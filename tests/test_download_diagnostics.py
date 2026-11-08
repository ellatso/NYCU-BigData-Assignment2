"""Offline download failures and resume contracts; never contact GH Archive."""
import contextlib
import ast
from collections import deque
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

spec=importlib.util.spec_from_file_location('prepare_revised',Path(__file__).resolve().parents[1]/'scripts/prepare_revised.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

class DownloadDiagnosticsTests(unittest.TestCase):
    def args(self,raw):
        return SimpleNamespace(raw=str(raw),start='2026-09-01T00:00:00+00:00',end='2026-09-01T02:00:00+00:00',storage='full',workers=1)

    def test_http_rejection_is_recorded_and_not_retried(self):
        with tempfile.TemporaryDirectory() as td:
            a=self.args(td);log=io.StringIO()
            error=HTTPError('https://data.gharchive.org/example',403,'Forbidden',{},io.BytesIO(b'error code: 1010'))
            with patch.object(prepare.urllib.request,'urlopen',side_effect=error) as request,contextlib.redirect_stdout(log):
                with self.assertRaises(SystemExit):prepare.download(a)
                self.assertEqual(request.call_count,1)
            record=next(iter(json.loads((Path(td)/'download_manifest.json').read_text())['files'].values()))
            self.assertEqual(record['http_status'],403)
            self.assertIn('1010',record['response_excerpt'])
            self.assertIn('Forbidden',log.getvalue())
            with patch.object(prepare.urllib.request,'urlopen') as request,contextlib.redirect_stdout(log):
                with self.assertRaisesRegex(SystemExit,'No further requests'):prepare.download(a)
                request.assert_not_called()
            self.assertFalse(list(Path(td).glob('.download_*')))

    def test_network_failure_returns_failure_and_preserves_reason(self):
        with tempfile.TemporaryDirectory() as td,patch.object(prepare.urllib.request,'urlopen',side_effect=URLError('fixture connection timed out')),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SystemExit,'source hours failed'):prepare.download(self.args(td))
            rows=json.loads((Path(td)/'download_manifest.json').read_text())['files']
            self.assertEqual(len(rows),2)
            self.assertTrue(all('timed out' in r['error'] for r in rows.values()))

    def test_verified_successes_resume_without_network(self):
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            a=self.args(td)
            with patch.object(prepare.urllib.request,'urlopen',side_effect=lambda *args,**kwargs:io.BytesIO(gzip.compress(b''))) as request:
                prepare.download(a);self.assertEqual(request.call_count,2)
            with patch.object(prepare.urllib.request,'urlopen') as request:
                prepare.download(a);request.assert_not_called()

    def test_notebook_keeps_child_stderr_in_log_and_exception(self):
        notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/Asg2_Revised_113700058.ipynb').read_text())
        cell=ast.parse(''.join(notebook['cells'][3]['source']))
        function=next(n for n in cell.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td)
            scope=dict(WORK=root,RAW=root/'raw',sys=sys,deque=deque,subprocess=subprocess,json=json)
            exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook-run>','exec'),scope)
            with self.assertRaisesRegex(RuntimeError,'fixture real cause'):
                scope['run']('-c',"import sys; print('fixture real cause',file=sys.stderr); sys.exit(3)")
            self.assertIn('fixture real cause',(root/'logs/latest_command.log').read_text())

if __name__=='__main__':unittest.main()
