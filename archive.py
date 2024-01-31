import datetime
import os
from shutil import copytree, ignore_patterns, rmtree

time = datetime.datetime.now()

version_dir = os.path.join('archive', time.strftime('%Y%m%d%H%M%S'))

print(f'Archiving to {version_dir}...')
copytree('./', version_dir, ignore=ignore_patterns('__pycache__', 'archive*'))

print('Cleaning output folder...')
if os.path.exists('output'):
    rmtree('output/')
else:
    print('No output folder found. Ignore.')

print(f'Done.')
