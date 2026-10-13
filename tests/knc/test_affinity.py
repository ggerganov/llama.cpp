#!/usr/bin/env python3
"""Test affinity policy with synthetic topology; no hardware qualification."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'knc-affinity.sh'


class AffinityPolicy(unittest.TestCase):
    def invoke(self, cores, siblings, interleaved=False, allowed=None, machine='k1om'):
        groups = [
            [core + thread * cores if interleaved else core * 4 + thread for thread in range(4)]
            for core in range(cores)
        ]
        with tempfile.TemporaryDirectory(prefix='knc-affinity-') as directory:
            for command, variable in [('uname', 'TEST_MACHINE'), ('cat', 'TEST_GROUPS'), ('awk', 'TEST_ALLOWED')]:
                wrapper = Path(directory) / command
                wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$' + variable + '"\n', encoding='utf-8')
                wrapper.chmod(0o700)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'], TEST_MACHINE=machine,
                       TEST_GROUPS='\n'.join(','.join(map(str, group)) for group in reversed(groups)),
                       TEST_ALLOWED=allowed if allowed is not None else f'0-{cores * 4 - 1}')
            result = subprocess.run(['bash', str(SCRIPT), str(siblings), '/usr/bin/env'], env=env,
                                    text=True, capture_output=True, check=False)
        return result, groups

    def test_all_cards_and_sibling_counts(self):
        for cores in (57, 60, 61):
            for interleaved in (False, True):
                for siblings in (1, 2, 3, 4):
                    with self.subTest(cores=cores, interleaved=interleaved, siblings=siblings):
                        result, groups = self.invoke(cores, siblings, interleaved)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        env = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
                        expected = [groups[core][thread] for thread in range(siblings) for core in range(1, cores)]
                        self.assertEqual(list(map(int, env['GOMP_CPU_AFFINITY'].split())), expected)
                        self.assertEqual(int(env['OMP_NUM_THREADS']), len(expected))
                        self.assertEqual(env['OMP_PROC_BIND'], 'TRUE')
                        self.assertEqual(env['OMP_DYNAMIC'], 'FALSE')

    def test_reject_excluded_worker(self):
        result, _ = self.invoke(61, 3, allowed='5-243')
        self.assertNotEqual(result.returncode, 0)

    def test_reject_non_knc_and_unknown_topology(self):
        for cores, machine in [(61, 'x86_64'), (58, 'k1om')]:
            result, _ = self.invoke(cores, 3, machine=machine)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
