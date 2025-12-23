# Task ID: 27

**Title:** Implement Structured Logging for ML Service

**Status:** done

**Dependencies:** None

**Priority:** medium

**Description:** Add JSON structured logging to the FastAPI ML service with correlation IDs, proper log levels, and sensitive data redaction.

**Details:**

**Update `ml-service/requirements.txt`:**
```
structlog>=24.1.0
python-json-logger>=2.0.7
```

**Create `ml-service/app/core/logging.py`:**
```python
import structlog
import logging
import sys
import uuid
from typing import Any
from contextvars import ContextVar

# Correlation ID context
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')

def get_correlation_id() -> str:
    return correlation_id_var.get() or str(uuid.uuid4())

def set_correlation_id(correlation_id: str) -> None:
    correlation_id_var.set(correlation_id)

# Sensitive data redaction
SENSITIVE_KEYS = {'password', 'token', 'secret', 'authorization', 'api_key'}

def redact_sensitive_data(_, __, event_dict: dict) -> dict:
    def redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: '[REDACTED]' if k.lower() in SENSITIVE_KEYS else redact(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [redact(item) for item in obj]
        return obj
    
    return redact(event_dict)

def add_service_context(_, __, event_dict: dict) -> dict:
    event_dict['service'] = 'nutri-ml-service'
    event_dict['correlation_id'] = get_correlation_id()
    return event_dict

def configure_logging(environment: str = 'development') -> None:
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        add_service_context,
        redact_sensitive_data,
    ]
    
    if environment == 'production':
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)
```

**Create `ml-service/app/middleware/logging.py`:**
```python
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import get_logger, set_correlation_id, get_correlation_id

logger = get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get('x-correlation-id', str(uuid.uuid4()))
        set_correlation_id(correlation_id)
        
        start_time = time.time()
        
        # Log request
        logger.info(
            'request_started',
            method=request.method,
            path=request.url.path,
            query=str(request.query_params),
        )
        
        try:
            response = await call_next(request)
            
            # Add correlation ID to response
            response.headers['x-correlation-id'] = correlation_id
            
            # Log response
            duration_ms = (time.time() - start_time) * 1000
            log_level = 'error' if response.status_code >= 500 else 'warn' if response.status_code >= 400 else 'info'
            getattr(logger, log_level)(
                'request_completed',
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            
            return response
        except Exception as e:
            logger.exception(
                'request_failed',
                method=request.method,
                path=request.url.path,
                error=str(e),
            )
            raise
```

**Update `ml-service/app/main.py`:**
```python
from app.core.logging import configure_logging, get_logger
from app.middleware.logging import LoggingMiddleware
from app.config import settings

# Configure logging on startup
configure_logging(settings.ENVIRONMENT)
logger = get_logger(__name__)

app = FastAPI(...)
app.add_middleware(LoggingMiddleware)

@app.on_event('startup')
async def startup():
    logger.info('ml_service_started', port=8000, environment=settings.ENVIRONMENT)
```

**Test Strategy:**

1. Start ML service and verify JSON log output in production mode
2. Verify correlation ID passed from backend appears in ML logs
3. Test sensitive data redaction in request logs
4. Verify log levels (debug, info, warn, error)
5. Test exception logging includes stack trace
6. Verify correlation ID in response headers
7. Test dev console renderer in development mode
8. Verify logs parse correctly with jq

## Subtasks

### 27.1. Install dependencies and create core logging module

**Status:** pending  
**Dependencies:** None  

Add structlog and python-json-logger to requirements.txt, then create app/core/logging.py with structured logging configuration, correlation ID context variables, and sensitive data redaction.

**Details:**

1. Update ml-service/requirements.txt with structlog>=24.1.0 and python-json-logger>=2.0.7
2. Create ml-service/app/core/logging.py implementing:
   - ContextVar for correlation IDs (correlation_id_var)
   - get_correlation_id() and set_correlation_id() functions
   - SENSITIVE_KEYS set for redaction (password, token, secret, authorization, api_key)
   - redact_sensitive_data() processor that recursively redacts dict/list values
   - add_service_context() processor adding service name and correlation_id
   - configure_logging() with environment-based processors (JSONRenderer for prod, ConsoleRenderer for dev)
   - get_logger() factory function
3. Install dependencies: cd ml-service && pip install -r requirements.txt
4. Test logging configuration imports without errors

### 27.2. Create logging middleware for FastAPI

**Status:** pending  
**Dependencies:** 27.1  

Implement FastAPI middleware in app/middleware/logging.py to capture request/response with timing metrics and propagate correlation IDs through headers.

**Details:**

1. Create ml-service/app/middleware/ directory if not exists
2. Create ml-service/app/middleware/logging.py implementing LoggingMiddleware(BaseHTTPMiddleware):
   - Extract or generate correlation ID from x-correlation-id header
   - Call set_correlation_id() to set context
   - Log request_started with method, path, query params
   - Measure request duration using time.time()
   - Add x-correlation-id to response headers
   - Log request_completed with status_code and duration_ms
   - Use dynamic log level based on status code (error >=500, warn >=400, info otherwise)
   - Log request_failed with exception details on errors
3. Test middleware with mock Request/Response objects

### 27.3. Integrate logging into FastAPI application

**Status:** pending  
**Dependencies:** 27.2  

Update ml-service/app/main.py to configure structured logging on startup and register the logging middleware.

**Details:**

1. Update ml-service/app/main.py imports:
   - from app.core.logging import configure_logging, get_logger
   - from app.middleware.logging import LoggingMiddleware
   - from app.config import settings (verify ENVIRONMENT exists in config)
2. Add configure_logging(settings.ENVIRONMENT) before app = FastAPI(...)
3. Create logger = get_logger(__name__) after configuration
4. Add app.add_middleware(LoggingMiddleware) after app instantiation
5. Update @app.on_event('startup') to log ml_service_started with port and environment
6. Start ML service and verify JSON logs in production mode: ENVIRONMENT=production uvicorn app.main:app
7. Verify console logs in dev mode: ENVIRONMENT=development uvicorn app.main:app

### 27.4. Replace existing logging calls with structured logger

**Status:** pending  
**Dependencies:** 27.3  

Audit and replace all existing Python logging.getLogger() calls throughout the ML service codebase with structlog get_logger(), ensuring consistent structured logging.

**Details:**

1. Find all existing logging usage: grep -r 'logging.getLogger\|logger.info\|logger.error\|logger.debug\|logger.warning' ml-service/app/ --include='*.py'
2. Replace logging imports with: from app.core.logging import get_logger
3. Replace logger = logging.getLogger(__name__) with logger = get_logger(__name__)
4. Update logging calls to use structured format:
   - Old: logger.info(f'Processing {metric_type}')
   - New: logger.info('processing_metric', metric_type=metric_type)
5. Focus on high-traffic files: app/main.py, app/routers/, app/services/, app/ml_models/
6. Add context to error logs: logger.error('operation_failed', error=str(e), user_id=user_id)
7. Test sensitive data redaction in actual service logs (password, token fields)
