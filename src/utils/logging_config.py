import logging

def setup_logging(debug=False):
    """
    Configure logging for the application.
    
    Args:
        debug: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()
