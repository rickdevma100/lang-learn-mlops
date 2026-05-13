def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests that load the full MLX model (~5 GB)"
    )
