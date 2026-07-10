import logging

def setup_logging(debug=False):
    """
    Configure logging for the application.
    
    Args:
        debug: If True, set log level to DEBUG, otherwise INFO
    """
    # Set root logger to a high level to suppress most messages
    logging.getLogger().setLevel(logging.WARNING)
    
    # Create our app logger
    app_logger = logging.getLogger('app')
    level = logging.DEBUG if debug else logging.INFO
    app_logger.setLevel(level)
    
    app_logger.propagate = False
    if not app_logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        app_logger.addHandler(console_handler)
    
    # Explicitly silence noisy libraries
    for noisy_logger in ['numba', 'matplotlib', 'PIL']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    return app_logger
