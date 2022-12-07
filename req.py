from pprint import pprint
path = '/root/work/jax-tutorial/requirements.txt'
new_path = '/root/work/jax-tutorial/req.txt'
with open(path) as f:
    lines = f.readlines()
for i in range(len(lines)):
    #もし@があったらそこから先を削除
    line = lines[i]
    if '@' in line:
        line = line[:line.index('@')-1]+'\n'
        lines[i] = line
    if '-e' in line:
        #削除
        lines[i] = ''
    #jupyterを削除
    if 'jupyter' in line:
        lines[i] = ''
    #conda関係を削除
    if 'conda' in line:
        lines[i] = ''

# fileを保存
with open(new_path, mode='w') as f:
    f.writelines(lines)