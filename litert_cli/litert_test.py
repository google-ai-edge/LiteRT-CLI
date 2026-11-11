# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import importlib
import json
import os
import subprocess
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import click
from click import testing
import huggingface_hub
from litert_cli import litert
from litert_cli.core import constants
import numpy as np


class LitertTest(parameterized.TestCase):

  def test_lazy_cli_list_commands(self) -> None:
    cli_group = litert.LazyCLI()
    commands = cli_group.list_commands(None)
    self.assertEqual(
        commands,
        [
            'benchmark',
            'clean',
            'compile',
            'convert',
            'delete',
            'download',
            'import',
            'list',
            'lm',
            'quantize',
            'run',
            'verify',
            'visualize',
        ],
    )

  @mock.patch.object(importlib, 'import_module', autospec=True)
  def test_lazy_cli_get_command(self, mock_import_module) -> None:
    # Mock the module and command object
    mock_mod = mock.create_autospec(types.ModuleType)
    mock_cmd = mock.create_autospec(click.Command)
    mock_mod.download_cmd = mock_cmd
    mock_import_module.return_value = mock_mod

    cli_group = litert.LazyCLI()
    cmd = cli_group.get_command(None, 'download')

    self.assertEqual(cmd, mock_cmd)
    mock_import_module.assert_called_once_with('litert_cli.commands.download')

  def test_lazy_cli_get_command_not_found(self) -> None:
    cli_group = litert.LazyCLI()
    cmd = cli_group.get_command(None, 'unknown_command')
    self.assertIsNone(cmd)

  @parameterized.parameters(
      'LiteRT CLI.',
      'download',
      'visualize',
      'clean',
      'compile',
      'convert',
      'lm',
      'quantize',
      'run',
  )
  def test_cli_help(self, expected_string) -> None:
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ['--help'])
    self.assertEqual(result.exit_code, 0)
    self.assertIn(expected_string, result.output)

  def test_run_reports_a_missing_model_file(self) -> None:
    runner = testing.CliRunner()
    result = runner.invoke(
        litert.cli, ['run', 'no_such_model.tflite', '--desktop', '--cpu']
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn('Model file not found: no_such_model.tflite', result.output)

  def test_run_rejects_a_directory(self) -> None:
    model_dir = self.create_tempdir().full_path
    result = testing.CliRunner().invoke(litert.cli, ['run', model_dir])
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn(f'Model file not found: {model_dir}', result.output)

  def test_run_rejects_an_out_of_range_signature_index(self) -> None:
    model = os.path.join(
        os.path.dirname(__file__), 'test_data', 'dummy_cv_model.tflite'
    )
    result = testing.CliRunner().invoke(
        litert.cli, ['run', model, '--desktop', '--signature-index', '5']
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn(
        '--signature-index 5 is out of range: the model has 1 signature(s)'
        " ('serving_default'), so use 0 to 0.",
        result.output,
    )


def _split_stream_runner() -> testing.CliRunner:
  """A runner that keeps stdout and stderr apart across click versions."""
  try:
    return testing.CliRunner(mix_stderr=False)  # click < 8.2
  except TypeError:
    return testing.CliRunner()  # click >= 8.2 never mixes them


_DUMMY_MODEL = os.path.join(
    os.path.dirname(__file__), 'test_data', 'dummy_cv_model.tflite'
)
# `scaled = x * 2 + 1` (float32[1, 4]) and `total = sum(x)` (scalar), with a
# float32 input `x`; and the same graph fully quantized with int8 I/O. See
# test_data/generate_verify_models.py.
_VERIFY_FLOAT_MODEL = os.path.join(
    os.path.dirname(__file__), 'test_data', 'verify_float.tflite'
)
_VERIFY_INT8_MODEL = os.path.join(
    os.path.dirname(__file__), 'test_data', 'verify_int8.tflite'
)


class VerifyTest(absltest.TestCase):
  """`litert verify` compares a model with reference outputs."""

  def test_npz_goldens_computed_outside_litert_pass(self) -> None:
    # Goldens computed with numpy (float64, as a script would produce them),
    # including a rank-0 output: this fails if the model is not actually run
    # or if scalar outputs come back with shape (1,).
    ref = self.create_tempdir().full_path
    x = np.array([[0.5, -1.0, 2.0, 0.25]])
    inputs_path = os.path.join(ref, 'inputs.npz')
    outputs_path = os.path.join(ref, 'outputs.npz')
    np.savez(inputs_path, x=x)
    np.savez(outputs_path, scaled=x * 2 + 1, total=np.sum(x))

    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _VERIFY_FLOAT_MODEL,
            '--inputs',
            inputs_path,
            '--expected',
            outputs_path,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    doc = json.loads(result.stdout)
    self.assertTrue(doc['passed'])
    by_name = {r['name']: r for r in doc['outputs']}
    self.assertEqual(by_name['scaled']['shape'], [1, 4])
    self.assertEqual(by_name['total']['shape'], [])
    self.assertEqual(by_name['total']['dtype'], 'float32')

    # Wrong goldens are detected, not masked by the scalar handling.
    np.savez(outputs_path, scaled=x * 2 + 1, total=np.sum(x) + 1.0)
    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _VERIFY_FLOAT_MODEL,
            '--inputs',
            inputs_path,
            '--expected',
            outputs_path,
        ],
    )
    self.assertEqual(result.exit_code, 5, result.output)
    by_name = {r['name']: r for r in json.loads(result.stdout)['outputs']}
    self.assertTrue(by_name['scaled']['passed'])
    self.assertFalse(by_name['total']['passed'])
    self.assertAlmostEqual(by_name['total']['max_abs_diff'], 1.0, places=5)

  def test_against_uses_given_inputs_and_saves_them(self) -> None:
    ref = self.create_tempdir().full_path
    x = np.array([[0.5, -1.0, 2.0, 0.25]], dtype=np.float32)
    inputs_path = os.path.join(ref, 'given.npz')
    np.savez(inputs_path, x=x)
    save_dir = os.path.join(ref, 'saved')

    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _VERIFY_FLOAT_MODEL,
            '--against',
            _VERIFY_FLOAT_MODEL,
            '--inputs',
            inputs_path,
            '--save',
            save_dir,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn(f'inputs from {inputs_path}', result.stderr)
    doc = json.loads(result.stdout)
    with np.load(doc['saved']['inputs']) as saved_inputs:
      np.testing.assert_array_equal(saved_inputs['x'], x)
    with np.load(doc['saved']['outputs']) as saved_outputs:
      np.testing.assert_allclose(saved_outputs['scaled'], x * 2 + 1)
      self.assertEqual(saved_outputs['total'].shape, ())

  def test_int8_model_against_float_original_is_rejected(self) -> None:
    result = _split_stream_runner().invoke(
        litert.cli,
        ['verify', _VERIFY_INT8_MODEL, '--against', _VERIFY_FLOAT_MODEL],
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Input 'x' has dtype float32", result.stderr)
    self.assertIn('expects int8', result.stderr)
    self.assertIn('not supported yet', result.stderr)

  def test_int8_model_against_itself_passes(self) -> None:
    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _VERIFY_INT8_MODEL,
            '--against',
            _VERIFY_INT8_MODEL,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    doc = json.loads(result.stdout)
    self.assertTrue(doc['passed'])
    self.assertEqual({r['dtype'] for r in doc['outputs']}, {'int8'})

  def test_against_itself_passes_and_saves_the_reference(self) -> None:
    ref = os.path.join(self.create_tempdir().full_path, 'ref')
    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _DUMMY_MODEL,
            '--against',
            _DUMMY_MODEL,
            '--save',
            ref,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    doc = json.loads(result.stdout)
    self.assertEqual(doc['schema'], 'litert.verify.v1')
    self.assertTrue(doc['passed'])
    self.assertEqual(doc['signature'], 'serving_default')
    for record in doc['outputs']:
      self.assertEqual(record['max_abs_diff'], 0.0)
    self.assertIn('Verification passed.', result.stderr)
    inputs = np.load(doc['saved']['inputs'])
    outputs = np.load(doc['saved']['outputs'])
    self.assertEqual(inputs.files, ['args_0'])
    self.assertEqual(outputs.files, ['output_0'])
    self.assertEqual(list(outputs['output_0'].shape), [1, 10])

  def test_npz_reference_and_tampering_exits_with_5(self) -> None:
    ref = os.path.join(self.create_tempdir().full_path, 'ref')
    runner = testing.CliRunner()
    result = runner.invoke(
        litert.cli,
        ['verify', _DUMMY_MODEL, '--against', _DUMMY_MODEL, '--save', ref],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    inputs_path = os.path.join(ref, 'inputs.npz')
    outputs_path = os.path.join(ref, 'outputs.npz')

    result = runner.invoke(
        litert.cli,
        [
            'verify',
            _DUMMY_MODEL,
            '--inputs',
            inputs_path,
            '--expected',
            outputs_path,
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)

    with np.load(outputs_path) as f:
      tampered = {k: f[k] + 1.0 for k in f.files}
    np.savez(outputs_path, **tampered)
    result = _split_stream_runner().invoke(
        litert.cli,
        [
            '--json',
            'verify',
            _DUMMY_MODEL,
            '--inputs',
            inputs_path,
            '--expected',
            outputs_path,
        ],
    )
    self.assertEqual(result.exit_code, 5, result.output)
    doc = json.loads(result.stdout)
    self.assertFalse(doc['passed'])
    self.assertIn('outside atol', doc['outputs'][0]['reason'])
    self.assertIn('Verification failed for', result.stderr)

    # Loosening the tolerance accepts the difference.
    result = runner.invoke(
        litert.cli,
        [
            'verify',
            _DUMMY_MODEL,
            '--inputs',
            inputs_path,
            '--expected',
            outputs_path,
            '--atol',
            '2',
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)

  def test_exactly_one_reference_is_required(self) -> None:
    runner = testing.CliRunner()
    result = runner.invoke(litert.cli, ['verify', _DUMMY_MODEL])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('exactly one reference', result.output)
    result = runner.invoke(
        litert.cli, ['verify', _DUMMY_MODEL, '--expected', _DUMMY_MODEL]
    )
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('--expected requires --inputs', result.output)

  def test_compare_outputs_reports_structural_mismatches(self) -> None:
    from litert_cli.commands import verify  # pylint: disable=g-import-not-at-top

    records = verify.compare_outputs(
        actual={
            'same': np.array([1.0, 2.0]),
            'shape': np.zeros((2, 2)),
            'extra': np.zeros(1),
        },
        expected={
            'same': np.array([1.0, 2.0 + 1e-7]),
            'shape': np.zeros(4),
            'missing': np.zeros(1),
        },
        atol=1e-5,
        rtol=1e-4,
    )
    by_name = {r['name']: r for r in records}
    self.assertTrue(by_name['same']['passed'])
    self.assertEqual(
        by_name['shape']['reason'], 'shape [2, 2] != reference [4]'
    )
    self.assertEqual(by_name['extra']['reason'], 'not in reference')
    self.assertEqual(by_name['missing']['reason'], 'missing from model outputs')

  def test_compare_outputs_keeps_nan_mismatches_out_of_the_stats(self) -> None:
    from litert_cli.commands import verify  # pylint: disable=g-import-not-at-top

    nan, inf = float('nan'), float('inf')
    records = verify.compare_outputs(
        # Matching nan/inf pairs pass and do not poison the stats.
        actual={
            'same': np.array([nan, inf, 1.0]),
            'nan_vs_finite': np.array([nan, 1.0]),
            'finite_vs_inf': np.array([1.0, 2.0]),
        },
        expected={
            'same': np.array([nan, inf, 1.0]),
            'nan_vs_finite': np.array([0.0, 1.0]),
            'finite_vs_inf': np.array([1.0, inf]),
        },
        atol=1e-5,
        rtol=1e-4,
    )
    by_name = {r['name']: r for r in records}
    self.assertTrue(by_name['same']['passed'])
    self.assertEqual(by_name['same']['max_abs_diff'], 0.0)
    self.assertFalse(by_name['nan_vs_finite']['passed'])
    self.assertEqual(by_name['nan_vs_finite']['mismatched'], 1)
    self.assertTrue(np.isnan(by_name['nan_vs_finite']['max_abs_diff']))
    self.assertFalse(by_name['finite_vs_inf']['passed'])
    self.assertEqual(by_name['finite_vs_inf']['max_abs_diff'], inf)

    # Under --json the non-finite stats become null, never bare NaN/Infinity.
    from litert_cli.core import report  # pylint: disable=g-import-not-at-top

    text = report.dumps({'outputs': records})
    self.assertNotIn('NaN', text)
    self.assertNotIn('Infinity', text)
    by_name = {r['name']: r for r in json.loads(text)['outputs']}
    self.assertIsNone(by_name['nan_vs_finite']['max_abs_diff'])
    self.assertIsNone(by_name['finite_vs_inf']['max_abs_diff'])


class EnvironmentErrorTest(parameterized.TestCase):

  @parameterized.parameters(
      (
          subprocess.CalledProcessError(
              1, ['adb', 'push'], stderr=b'no device'
          ),
          '`adb push` exited with 1: no device',
      ),
      (OSError(28, 'No space left on device'), 'No space left on device'),
  )
  def test_environment_errors_have_no_traceback(self, error, message) -> None:
    with mock.patch.object(
        litert.LazyCLI, 'get_command', autospec=True
    ) as get_command:

      @click.command('boom')
      def boom():
        raise error

      get_command.return_value = boom
      result = testing.CliRunner().invoke(litert.cli, ['boom'])
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn(message, result.output)
    self.assertIn('LITERT_VERBOSE=1', result.output)
    self.assertIsInstance(result.exception, SystemExit)


class ConvertOptionsTest(parameterized.TestCase):
  """Options of one conversion mode are rejected in the other."""

  def setUp(self):
    super().setUp()
    self.enter_context(mock.patch.object(constants, 'IS_INTERNAL_ENV', True))

  @parameterized.parameters(
      (['my_model.py', '--cache-length', '1024'], '--cache-length'),
      (['my_model.py', '--dry-run'], '--dry-run'),
      (['org/model', '--model-args', 'a=1'], '--model-args'),
      (['org/model', '--export-aipack', 'pack'], '--export-aipack'),
      (['my_model.py', '--export-aipack', 'pack'], 'needs --target'),
  )
  def test_options_of_the_other_mode_are_errors(self, args, message) -> None:
    result = testing.CliRunner().invoke(litert.cli, ['convert', *args])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn(message, result.output)

  def test_prefill_lengths_must_be_positive_integers(self) -> None:
    result = testing.CliRunner().invoke(
        litert.cli, ['convert', 'org/model', '--prefill-lengths', '128,x']
    )
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('positive integers', result.output)

  def test_model_args_are_python_literals(self) -> None:
    from litert_cli.commands.convert import generic  # pylint: disable=g-import-not-at-top

    self.assertEqual(
        generic.parse_model_args('a=4,b=-1,c=1e-3,d=true,e=x'),
        {'a': 4, 'b': -1, 'c': 1e-3, 'd': True, 'e': 'x'},
    )
    with self.assertRaisesRegex(click.BadParameter, 'key=value'):
      generic.parse_model_args('batch_size')

  def test_unknown_script_recipe_is_an_error(self) -> None:
    from litert_cli.commands.convert import generic  # pylint: disable=g-import-not-at-top

    self.assertIsNone(generic._quant_config_for_recipe('none'))
    with self.assertRaisesRegex(click.BadParameter, 'weight_only_wi4_afp32'):
      generic._quant_config_for_recipe('weight_only_w4')


class QuantizeRecipesTest(absltest.TestCase):

  def test_only_public_recipe_functions_are_listed(self) -> None:
    from litert_cli.commands import quantize  # pylint: disable=g-import-not-at-top

    module = types.ModuleType('fake_recipe')
    exec(  # pylint: disable=exec-used
        'import json\n'
        'def dynamic_wi8_afp32(): return 1\n'
        'def _private(): return 2\n',
        module.__dict__,
    )
    self.assertEqual(
        list(quantize._builtin_recipes(module)), ['dynamic_wi8_afp32']
    )


class LmHelpTest(absltest.TestCase):

  def test_help_starts_with_the_litert_cli_options(self) -> None:
    with (
        mock.patch.object(constants, 'IS_INTERNAL_ENV', True),
        mock.patch('subprocess.run', autospec=True) as run,
    ):
      run.return_value = mock.Mock(returncode=0)
      result = testing.CliRunner().invoke(litert.cli, ['lm', '--help'])
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('--reconvert', result.output)
    run.assert_called_once_with(['litert-lm', '--help'], check=False)


class ModelCacheCommandsTest(parameterized.TestCase):
  """`delete`, `import` and `clean` must stay inside the CLI cache."""

  def setUp(self):
    super().setUp()
    self.cache = self.create_tempdir('cache').full_path
    self.models = self.create_tempdir('cache/models').full_path
    self.enter_context(
        mock.patch.object(constants, 'LITERT_CLI_CACHE_DIR', self.cache)
    )
    self.enter_context(
        mock.patch.object(constants, 'LITERT_MODELS_CACHE_DIR', self.models)
    )
    self.enter_context(
        mock.patch.object(
            constants, 'LITERT_CLI_ROOT', os.path.join(self.cache, 'root')
        )
    )

  @parameterized.parameters('..', '.', '', 'a\\b')
  def test_delete_rejects_refs_outside_the_models_cache(self, ref) -> None:
    result = testing.CliRunner().invoke(litert.cli, ['delete', ref, '-y'])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('not a valid model reference name', result.output)
    self.assertTrue(os.path.isdir(self.models))

  def test_delete_removes_a_flattened_ref(self) -> None:
    os.makedirs(os.path.join(self.models, 'org__name'))
    result = testing.CliRunner().invoke(
        litert.cli, ['delete', 'org/name', '-y']
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertFalse(os.path.exists(os.path.join(self.models, 'org__name')))

  def test_delete_rejects_a_sub_reference(self) -> None:
    os.makedirs(os.path.join(self.models, 'foo'))
    result = testing.CliRunner().invoke(litert.cli, ['delete', 'foo:enc', '-y'])
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertIn('sub-reference', result.output)
    self.assertTrue(os.path.isdir(os.path.join(self.models, 'foo')))

  def test_list_shows_a_model(self) -> None:
    os.makedirs(os.path.join(self.models, 'org__name'))
    with open(os.path.join(self.models, 'org__name', 'm.tflite'), 'w') as f:
      f.write('x')
    result = testing.CliRunner().invoke(litert.cli, ['list', 'org/name'])
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('m.tflite', result.output)

  def test_import_rejects_a_ref_outside_the_models_cache(self) -> None:
    model = self.create_tempfile('m.tflite', content='x').full_path
    result = testing.CliRunner().invoke(
        litert.cli, ['import', model, '--model-ref', '..']
    )
    self.assertEqual(result.exit_code, 2, result.output)
    self.assertFalse(os.path.exists(os.path.join(self.cache, 'm.tflite')))

  def test_clean_asks_before_deleting(self) -> None:
    result = testing.CliRunner().invoke(litert.cli, ['clean'], input='n\n')
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('every managed model', result.output)
    self.assertIn('Aborted.', result.output)
    self.assertTrue(os.path.isdir(self.models))


class DownloadTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.list_repo_files = self.enter_context(
        mock.patch.object(
            huggingface_hub.HfApi,
            'list_repo_files',
            autospec=True,
            return_value=['README.md', 'a.litertlm', 'b.litertlm'],
        )
    )
    self.snapshot_download = self.enter_context(
        mock.patch.object(
            huggingface_hub, 'snapshot_download', autospec=True
        )
    )
    self.output = self.create_tempdir().full_path
    self.snapshot_download.return_value = self.output

  def test_file_that_matches_nothing_is_an_error(self) -> None:
    result = testing.CliRunner().invoke(
        litert.cli,
        ['download', 'org/name', '--file', 'gone.litertlm', '--output',
         self.output],
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn(
        "No file in org/name matches --file 'gone.litertlm'. Files in the"
        ' repo: README.md, a.litertlm, b.litertlm',
        result.output,
    )
    self.snapshot_download.assert_not_called()

  def test_failed_managed_download_leaves_no_model(self) -> None:
    cache = self.create_tempdir().full_path
    models_dir = os.path.join(cache, 'models')
    with mock.patch.object(constants, 'LITERT_MODELS_CACHE_DIR', models_dir):
      result = testing.CliRunner().invoke(
          litert.cli,
          ['download', 'org/name', '--file', 'gone.litertlm', '--model-ref',
           'mine'],
      )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertFalse(os.path.exists(os.path.join(models_dir, 'mine')))

  def test_file_pattern_that_matches_is_downloaded(self) -> None:
    result = testing.CliRunner().invoke(
        litert.cli,
        ['download', 'org/name', '--file', '*.litertlm', '--output',
         self.output],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.snapshot_download.assert_called_once()
    kwargs = self.snapshot_download.call_args.kwargs
    self.assertEqual(kwargs['repo_id'], 'org/name')
    self.assertEqual(kwargs['allow_patterns'], '*.litertlm')


if __name__ == '__main__':
  absltest.main()
