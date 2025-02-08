import { useEffect, useState } from 'react';
import { useAppContext } from '../utils/app.context';
import { XCloseButton } from '../utils/common';
import { delay } from '../utils/misc';
import StorageUtils from '../utils/storage';
import { CanvasType } from '../utils/types';
import { PlayIcon } from '@heroicons/react/24/outline';

const PyodideWrapper = {
  load: async function () {
    // load pyodide from CDN
    // @ts-expect-error experimental pyodide
    if (window.addedScriptPyodide) return;
    // @ts-expect-error experimental pyodide
    window.addedScriptPyodide = true;
    const scriptElem = document.createElement('script');
    scriptElem.src = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js';
    document.body.appendChild(scriptElem);
  },

  run: async function (code: string) {
    PyodideWrapper.load();

    // wait for pyodide to be loaded
    // @ts-expect-error experimental pyodide
    while (!window.loadPyodide) {
      await delay(100);
    }
    const stdOutAndErr: string[] = [];
    // @ts-expect-error experimental pyodide
    const pyodide = await window.loadPyodide({
      stdout: (data: string) => stdOutAndErr.push(data),
      stderr: (data: string) => stdOutAndErr.push(data),
    });
    const result = await pyodide.runPythonAsync(code);
    if (result) {
      stdOutAndErr.push(result.toString());
    }
    return stdOutAndErr.join('');
  },
};

if (StorageUtils.getConfig().pyIntepreterEnabled) {
  PyodideWrapper.load();
}

export default function CanvasPyInterpreter() {
  const { canvasData, setCanvasData } = useAppContext();

  const [running, setRunning] = useState(false);
  const [output, setOutput] = useState('');

  const runCode = async () => {
    const code = canvasData?.content;
    if (!code) return;
    setRunning(true);
    setOutput('Running...');
    const out = await PyodideWrapper.run(code);
    setOutput(out);
    setRunning(false);
  };

  // run code on mount
  useEffect(() => {
    runCode();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (canvasData?.type !== CanvasType.PY_INTERPRETER) {
    return null;
  }

  return (
    <div className="card bg-base-200 w-full h-full shadow-xl">
      <div className="card-body">
        <div className="flex justify-between items-center mb-4">
          <span className="text-lg font-bold">Pyodide</span>
          <XCloseButton
            className="bg-base-100"
            onClick={() => setCanvasData(null)}
          />
        </div>
        <div className="grid grid-rows-3 gap-4 h-full">
          <textarea
            className="textarea textarea-bordered w-full h-full font-mono"
            value={canvasData.content}
            onChange={(e) =>
              setCanvasData({
                ...canvasData,
                content: e.target.value,
              })
            }
          ></textarea>
          <div className="font-mono flex flex-col row-span-2">
            <div className="flex items-center mb-2">
              <button
                className="btn btn-sm bg-base-100"
                onClick={runCode}
                disabled={running}
              >
                <PlayIcon className="h-6 w-6" />{' '}
                {running ? 'Running...' : 'Run'}
              </button>
            </div>
            <pre className="bg-slate-900 rounded-md grow text-gray-200 p-3">
              {output}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
