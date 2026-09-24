import ast
from pathlib import Path

base = Path(r'c:\Users\PC\Desktop\ADVANCE AI\WEEK 11 AUTOMATION')
for fname in ['app.py', 'NIA26AI016_WEE11_VIDEO3.py']:
    source = (base / fname).read_text(encoding='utf-8')
    mod = ast.parse(source)
    ns = {}
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'chunk_text', 'rag_search'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), fname, 'exec'), ns)

    text = 'This is sentence one. ' * 50
    chunks = ns['chunk_text'](text, chunk_size=100, overlap=20)
    assert chunks and all(len(c) > 20 for c in chunks), f'{fname}: chunking invalid'

    class Result:
        def __init__(self, data):
            self.data = data

    class FakeSupabase:
        def rpc(self, name, payload):
            if name == 'match_documents':
                return Result([{'title': 'demo', 'similarity': 0.9}])
            raise RuntimeError('wrong rpc')

    ns['supabase'] = FakeSupabase()
    result = ns['rag_search']('What is the policy?', top_k=3)
    assert result and result[0]['title'] == 'demo', f'{fname}: rag search failed'
    print(f'{fname}: OK -> {len(chunks)} chunks, {len(result)} matches')
