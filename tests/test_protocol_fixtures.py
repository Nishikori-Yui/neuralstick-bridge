"""Golden wire frames validate against schema and the actual native/client paths."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from ncs2_client import NCS2Worker
from ncs2_multi_client import NCS2MultiWorker

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'tests/worker_protocol_fixtures.json'


class ProtocolFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(CORPUS.read_text())['cases']

    def test_json_schema_matches_all_golden_frames(self):
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest('Install the [dev] extra to validate JSON Schema')
        schema = json.loads((ROOT/'docs/worker-protocol.schema.json').read_text())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for case in self.cases:
            with self.subTest(case=case['name']):
                self.assertEqual(validator.is_valid(case['frame']),case['valid'])

    def test_python_clients_consume_golden_ready_and_output(self):
        for version, client_type in [(1,NCS2Worker),(2,NCS2MultiWorker)]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as d:
                worker=Path(d)/'worker'
                script = '''import json,sys
cases=json.load(open(sys.argv[1]))['cases']
frames={c['name']:c['frame'] for c in cases}
prefix='schemaVERSION-'
print(json.dumps(frames[prefix+'ready']),flush=True)
for line in sys.stdin:
 req=json.loads(line)
 key='infer-response' if req['operation']=='infer' else 'control-response'
 response=dict(frames[prefix+key],request_id=req['request_id'])
 print(json.dumps(response),flush=True)
 if req['operation']=='shutdown': break
'''.replace('VERSION',str(version)).replace("sys.argv[1]",repr(str(CORPUS)))
                worker.write_text('#!/usr/bin/env python3\n'+script)
                worker.chmod(0o700)
                frames={c['name']:c['frame'] for c in self.cases}
                req=frames[f'schema{version}-infer-request']
                inputs=req['input']['data'] if version==1 else [t['data'] for t in req['inputs']]
                with client_type(worker,Path('/unused'),Path('/unused/fixture.xml')) as client:
                    value=client.infer(inputs)
                    self.assertTrue(value['hardware_executed'])
                    self.assertEqual(value['output'] if version==1 else value['outputs'][0]['data'],[0,0.5,0,3])

    def test_native_decoders_accept_shared_requests(self):
        for version, name in [(1,'ncs2_worker'),(2,'ncs2_multi_worker')]:
            worker=ROOT/'build-arm64'/name
            if not worker.exists():
                self.skipTest('Build workers with ./ncs2 build-probe')
            cases=[c['frame'] for c in self.cases if c['name'].startswith(f'schema{version}-') and c['valid'] and c['name'].endswith('-request')]
            cases.sort(key=lambda f: f['operation']=='shutdown')
            result=subprocess.run([str(worker),'--validate-only'],input=''.join(json.dumps(f)+'\n' for f in cases),capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            responses=[json.loads(line) for line in result.stdout.splitlines()][1:]
            self.assertEqual(len(responses),len(cases))
            self.assertTrue(all(r['status']=='ok' for r in responses))
