"""Bundle this package explicitly so exported FL clients do not need the checkout."""
from pathlib import Path


def include_runtime_sources(job, client_names=None):
    package = Path(__file__).resolve().parent
    for source in sorted(package.rglob('*.py')):
        relative_parent = source.relative_to(package.parent).parent.as_posix()
        job.add_file_to_server(str(source), dest_dir=relative_parent)
        if client_names is None:
            job.add_file_to_clients(str(source), dest_dir=relative_parent)
        else:
            for site in client_names:
                job.add_file_to(str(source), site, dest_dir=relative_parent)
