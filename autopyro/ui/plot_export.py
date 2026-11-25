"""Process lifecycle and Qt notifications for one export job at a time."""
import multiprocessing
import os
import re
from pathlib import Path
import tempfile

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QApplication

from autopyro.model.core import timestamp
from autopyro.data_io.file_output import write_json
from autopyro.visualization.plot_export import export_files, with_export_style
from autopyro.visualization.plotting import PlotSpec
from .i18n import tr


def plan_batch_export(plots, directory, extension, *, names=None, style=None):
    """Prepare ordered jobs and choose available output paths without writing files."""
    descriptions = []
    chosen = plots.items() if names is None else ((name, plots[name]) for name in names if name in plots)
    for i, (name, config) in enumerate(chosen, start=1):
        readable = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:80].strip(' .')
        stem = f'plot-{i:03d}' + (f'-{readable}' if readable else '')
        path = Path(directory) / f'{stem}.{extension}'
        suffix = 2
        while path.exists():
            path = Path(directory) / f'{stem}-{suffix}.{extension}'
            suffix += 1
        try:
            spec = PlotSpec.from_dict(config)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(tr('Недопустимая конфигурация {0}: {1}', name, str(exc))) from exc
        descriptions.append((name, path, with_export_style(spec, style)))
    manifest = Path(directory) / 'plots-manifest.json'
    suffix = 2
    while manifest.exists():
        manifest = Path(directory) / f'plots-manifest-{suffix}.json'
        suffix += 1
    return descriptions, manifest


class _ExportWorker(QThread):
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, jobs, manifest, parent):
        super().__init__(parent)
        self.jobs, self.manifest = jobs, manifest
        self.context = multiprocessing.get_context('spawn')
        self.cancelled = self.context.Event()
        self.process = None

    def run(self):
        stages = []
        receiver = sender = None
        try:
            for job in self.jobs:
                path = Path(job['path'])
                fd, staging = tempfile.mkstemp(dir=path.parent, prefix='.' + path.stem + '-export-', suffix=path.suffix)
                os.close(fd)
                stages.append(Path(staging))
                job['staging'] = staging
                job['no_overwrite'] = self.manifest is not None
            receiver, sender = self.context.Pipe(duplex=False)
            self.process = self.context.Process(target=export_files, args=(self.jobs, self.cancelled, sender))
            if self.cancelled.is_set():
                outcome = {'plots': [], 'errors': [], 'cancelled': True}
                if self.manifest:
                    write_json(self.manifest, {'created_at': timestamp(), **outcome}, overwrite=False)
                self.succeeded.emit(outcome)
                return
            # Spawn and serialization are deliberately off the GUI thread.
            self.process.start()
            sender.close()
            outcome = None
            while True:
                try:
                    available = receiver.poll(.05)
                except (EOFError, OSError):
                    break
                if available:
                    try:
                        item = receiver.recv()
                    except (EOFError, OSError):
                        break
                    if 'progress' in item:
                        self.progress.emit(item['progress'], item['total'])
                    elif 'result' in item:
                        outcome = item['result']
                    elif 'fatal' in item:
                        raise RuntimeError(item['fatal'])
                elif not self.process.is_alive():
                    break
            self.process.join()
            if outcome is None:
                raise RuntimeError(f'Процесс экспорта завершился без результата: {self.process.exitcode}')
            if self.manifest:
                write_json(self.manifest, {'created_at': timestamp(), **outcome}, overwrite=False)
            self.succeeded.emit(outcome)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if self.process is not None and self.process.pid is not None:
                if self.process.is_alive():
                    self.process.terminate()
                self.process.join()
            for connection in (receiver, sender):
                if connection is not None:
                    connection.close()
            for staging in stages:
                staging.unlink(missing_ok=True)
            self.jobs = None


class PlotExport(QObject):
    progress = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)
    busyChanged = Signal(bool)

    def __init__(self):
        super().__init__(QApplication.instance())
        self.worker = None
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def start(self, jobs, manifest=None):
        if self.worker is not None:
            return False
        worker = self.worker = _ExportWorker(jobs, manifest, self)
        outcome = {}
        worker.progress.connect(self.progress)
        worker.succeeded.connect(lambda value: outcome.update(result=value))
        worker.failed.connect(lambda error: outcome.update(error=error))
        def finished():
            self.worker = None
            self.busyChanged.emit(False)
            if 'result' in outcome:
                self.completed.emit(outcome['result'])
            elif 'error' in outcome:
                self.failed.emit(outcome['error'])
            outcome.clear()
            worker.deleteLater()
        worker.finished.connect(finished)
        self.busyChanged.emit(True)
        worker.start()
        return True

    def cancel(self):
        if self.worker is not None:
            self.worker.cancelled.set()

    def shutdown(self):
        self.cancel()
        if self.worker is not None and not self.worker.wait(2000):
            process = self.worker.process
            if process is not None and process.pid is not None and process.is_alive():
                process.terminate()
            self.worker.wait()
