def test_import():
    import importlib
    mod = importlib.import_module("microbrain")
    assert mod is not None
