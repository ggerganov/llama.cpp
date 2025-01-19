#!/bin/env python3
import _io
import re
import os
import sys
from datetime import date
from pathlib import Path
from urllib import request
from urllib.request import urlopen

reUrl = re.compile('^(http(s|)://)(www.|)[a-zA-Z0-9.]*/.*$')
reSupportedIsas = re.compile('.*static constexpr Isa supportedIsas_.*')
reTarget = re.compile('.*{([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)},.*')

src = "https://raw.githubusercontent.com/ROCm/clr/refs/heads/amd-staging/rocclr/device/device.cpp"
srcType = 'url'

targets = []

def parse(items):
    assert(type(items) == list )

    depth = 0
    i = 0
    for line in items:
        i += 1
        line = str(line.encode("utf-8"))

        if re.match(reSupportedIsas, line):
            depth += 1
            continue

        if depth:
            for char in line:
                if char == '}':
                    depth -= 1
                    if depth < 1:
                        break
                elif char == '{':
                    depth += 1

            if depth < 1:
                break

            if re.match(reTarget, line):
                itms = reTarget.split(line)
                targets.append((itms[1].strip(' "'),itms[5].strip(' '),itms[6].strip(' '),itms[7].strip(' ')))


if __name__ == '__main__':
    buffer=""

    if len(sys.argv) > 1:
        src = sys.argv[1]
        if re.fullmatch(reUrl, src):
            srcType = 'url'

        else:
            srcType = 'file'
            if not os.path.exists(src):
                raise FileNotFoundError

            _src = Path(src)
            if not _src.exists():
                raise FileNotFoundError

    if srcType == "url":
        urlreq = request.Request(src)
        data = urlopen(urlreq)
        buffer = str(data.read().decode("utf-8"))

        parse(buffer.splitlines())
    else:
        try:
            num_lines = -1
            with open(_src, 'r') as fileIn:
                buffer = fileIn.readlines()

            parse(buffer)

        except Exception as exception:
            print(exception)
        finally:
            if isinstance(fileIn, _io.TextIOWrapper) and not fileIn.close:
                fileIn.close()

    if len(targets) == 0:
        print(f'No items found in {src}!', file=sys.stderr)
        exit(1)

    i = 0
    print(f'struct target '"{")
    print(f'    char id[256];')
    print(f'    char major;')
    print(f'    char minor;')
    print(f'    char step;')
    print("};")
    print('')
    print(f'// Automatically generated on {date.today()} from "{src}"')
    print(f'struct target targets[{len(targets)}];')
    for itm in targets:
        assert(type(itm) == tuple)
        print(f'strcpy(targets[{i}].id, "{itm[0]}");')
        print(f'targets[{i}].major = {itm[1]};')
        print(f'targets[{i}].minor = {itm[2]};')
        print(f'targets[{i}].step  = {itm[3]};')
        i += 1
