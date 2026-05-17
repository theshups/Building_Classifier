"""logger.py - UTF-8 safe logging for Windows + Unix"""
import io, logging, os, sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

LOG_FILE      = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
LOGS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

_FMT = logging.Formatter(
    fmt="[%(asctime)s]  %(levelname)-8s  %(name)-28s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_h = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
_file_h.setFormatter(_FMT)

try:
    _stream_h = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                         errors="replace", line_buffering=True))
except AttributeError:
    _stream_h = logging.StreamHandler(sys.stdout)
_stream_h.setFormatter(_FMT)

logging.basicConfig(level=logging.INFO, handlers=[_file_h, _stream_h])
for _lib in ("urllib3","PIL","matplotlib","absl","h5py","tensorflow",
             "wandb","mlflow","roboflow","urllib"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logger = get_logger("property_classifier")
logger.info(f"Logger ready -> {LOG_FILE_PATH}")
