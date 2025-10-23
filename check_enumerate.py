import re
import glob

for f in glob.glob('**/*.tex', recursive=True):
    with open(f, encoding='utf-8') as file:
        content = file.read()
    begins = len(re.findall(r'\\begin\{enumerate\}', content))
    ends = len(re.findall(r'\\end\{enumerate\}', content))
    if begins != ends:
        print(f'{f}: {begins} begins, {ends} ends')
