import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class StepLogger:
    """Enhanced step logger with standard logging interface"""

    def __init__(self,
                 filepath: str,
                 log_level: str = "INFO",
                 enable_console: bool = True,
                 max_entries: int = 10000,
                 auto_rotate: bool = True):

        self.filepath = Path(filepath)
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.max_entries = max_entries
        self.auto_rotate = auto_rotate
        self.entry_count = 0

        # Ensure directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Set up standard logger
        self._setup_standard_logger()

        # Load existing entry count
        self._count_existing_entries()

    def _setup_standard_logger(self):
        """Set up standard Python logger"""
        self.logger = logging.getLogger(f"StepLogger_{self.filepath.name}")
        self.logger.setLevel(self.log_level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # File handler for step logs
        file_handler = logging.FileHandler(
            str(self.filepath.parent / f"{self.filepath.stem}_standard.log")
        )
        file_handler.setLevel(self.log_level)

        # Console handler if enabled
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)

            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _count_existing_entries(self):
        """Count existing entries in the log file"""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    self.entry_count = sum(1 for line in f if line.strip())
        except Exception as e:
            self.logger.error(f"Failed to count existing entries: {e}")
            self.entry_count = 0

    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds max entries"""
        if not self.auto_rotate or self.entry_count < self.max_entries:
            return

        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.filepath.parent / f"{self.filepath.stem}_backup_{timestamp}.jsonl"

            # Move current log to backup
            if self.filepath.exists():
                self.filepath.rename(backup_path)
                self.logger.info(f"Rotated log file to: {backup_path}")

            # Reset counter
            self.entry_count = 0

        except Exception as e:
            self.logger.error(f"Failed to rotate log file: {e}")

    def log_step(self, context: str, output: str, metadata: Optional[Dict[str, Any]] = None):
        """Original step logging method"""
        try:
            # Check if rotation is needed
            self._rotate_log_if_needed()

            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "context": context,
                "output": output,
                "metadata": metadata or {},
                "entry_type": "reasoning_step"
            }

            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.entry_count += 1
            self.logger.debug(f"Logged reasoning step {self.entry_count}")

        except Exception as e:
            self.logger.error(f"Failed to log step: {e}")

    def log_reasoning_trace(self,
                            original_query: str,
                            final_output: str,
                            steps: List[Dict[str, Any]],
                            metadata: Optional[Dict[str, Any]] = None):
        """Log complete reasoning trace"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "entry_type": "reasoning_trace",
                "original_query": original_query,
                "final_output": final_output,
                "steps": steps,
                "total_steps": len(steps),
                "metadata": metadata or {}
            }

            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.entry_count += 1
            self.logger.info(f"Logged complete reasoning trace with {len(steps)} steps")

        except Exception as e:
            self.logger.error(f"Failed to log reasoning trace: {e}")

    def log_error(self, error_message: str, context: str = "", metadata: Optional[Dict[str, Any]] = None):
        """Log error information"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "entry_type": "error",
                "error_message": error_message,
                "context": context,
                "metadata": metadata or {}
            }

            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.entry_count += 1
            self.logger.error(f"Error logged: {error_message}")

        except Exception as e:
            self.logger.error(f"Failed to log error: {e}")

    def log_performance(self,
                        operation: str,
                        duration_ms: int,
                        success: bool = True,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "entry_type": "performance",
                "operation": operation,
                "duration_ms": duration_ms,
                "success": success,
                "metadata": metadata or {}
            }

            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.entry_count += 1
            self.logger.debug(f"Performance logged: {operation} took {duration_ms}ms")

        except Exception as e:
            self.logger.error(f"Failed to log performance: {e}")

    # Standard logging interface methods
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(message)
        if extra:
            self._log_structured_message("debug", message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message)
        if extra:
            self._log_structured_message("info", message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message)
        if extra:
            self._log_structured_message("warning", message, extra)

    def warn(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Alias for warning"""
        self.warning(message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self.logger.error(message)
        if extra:
            self._log_structured_message("error", message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self.logger.critical(message)
        if extra:
            self._log_structured_message("critical", message, extra)

    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log exception with traceback"""
        self.logger.exception(message)
        if extra:
            self._log_structured_message("exception", message, extra)

    def _log_structured_message(self, level: str, message: str, extra: Dict[str, Any]):
        """Log structured message to JSON file"""
        try:
            entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "entry_type": "log_message",
                "level": level,
                "message": message,
                "extra": extra
            }

            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self.entry_count += 1

        except Exception as e:
            # Use standard logger to avoid recursion
            self.logger.error(f"Failed to log structured message: {e}")

    def get_recent_entries(self, count: int = 100, entry_type: str = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        try:
            entries = []

            if not self.filepath.exists():
                return entries

            with open(self.filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Get last 'count' lines
            recent_lines = lines[-count:] if len(lines) > count else lines

            for line in recent_lines:
                try:
                    entry = json.loads(line.strip())
                    if entry_type is None or entry.get('entry_type') == entry_type:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

            return entries

        except Exception as e:
            self.logger.error(f"Failed to get recent entries: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        try:
            stats = {
                "total_entries": self.entry_count,
                "log_file_path": str(self.filepath),
                "log_level": logging.getLevelName(self.log_level),
                "auto_rotate": self.auto_rotate,
                "max_entries": self.max_entries
            }

            # Get file size if exists
            if self.filepath.exists():
                stats["file_size_mb"] = round(self.filepath.stat().st_size / (1024 * 1024), 2)
            else:
                stats["file_size_mb"] = 0

            # Count entry types
            recent_entries = self.get_recent_entries(1000)  # Last 1000 entries
            entry_types = {}
            for entry in recent_entries:
                entry_type = entry.get('entry_type', 'unknown')
                entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

            stats["entry_types"] = entry_types

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def search_entries(self,
                       query: str = None,
                       entry_type: str = None,
                       start_time: str = None,
                       end_time: str = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search log entries with filters"""
        try:
            matching_entries = []

            if not self.filepath.exists():
                return matching_entries

            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())

                        # Filter by entry type
                        if entry_type and entry.get('entry_type') != entry_type:
                            continue

                        # Filter by time range
                        entry_time = entry.get('timestamp', '')
                        if start_time and entry_time < start_time:
                            continue
                        if end_time and entry_time > end_time:
                            continue

                        # Filter by query (search in context, output, message)
                        if query:
                            searchable_text = ' '.join([
                                entry.get('context', ''),
                                entry.get('output', ''),
                                entry.get('message', ''),
                                str(entry.get('metadata', {}))
                            ]).lower()

                            if query.lower() not in searchable_text:
                                continue

                        matching_entries.append(entry)

                        if len(matching_entries) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

            return matching_entries

        except Exception as e:
            self.logger.error(f"Failed to search entries: {e}")
            return []

    def clear_logs(self, keep_backup: bool = True) -> bool:
        """Clear log files"""
        try:
            if keep_backup and self.filepath.exists():
                # Create backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.filepath.parent / f"{self.filepath.stem}_cleared_backup_{timestamp}.jsonl"
                self.filepath.rename(backup_path)
                self.logger.info(f"Log cleared, backup created: {backup_path}")
            else:
                # Just remove the file
                if self.filepath.exists():
                    self.filepath.unlink()
                    self.logger.info("Log file cleared")

            self.entry_count = 0
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear logs: {e}")
            return False

    def export_logs(self,
                    output_path: str,
                    format: str = "json",
                    filters: Optional[Dict[str, Any]] = None) -> bool:
        """Export logs to different formats"""
        try:
            output_path = Path(output_path)
            filters = filters or {}

            # Get entries with filters
            entries = self.search_entries(
                query=filters.get('query'),
                entry_type=filters.get('entry_type'),
                start_time=filters.get('start_time'),
                end_time=filters.get('end_time'),
                limit=filters.get('limit', 10000)
            )

            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)

            elif format.lower() == "csv":
                import csv
                if entries:
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        # Get all possible field names
                        fieldnames = set()
                        for entry in entries:
                            fieldnames.update(entry.keys())
                        fieldnames = list(fieldnames)

                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for entry in entries:
                            # Convert complex fields to strings
                            row = {}
                            for key, value in entry.items():
                                if isinstance(value, (dict, list)):
                                    row[key] = json.dumps(value)
                                else:
                                    row[key] = value
                            writer.writerow(row)

            self.logger.info(f"Exported {len(entries)} entries to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            return False


# Factory function for easy creation
def create_step_logger(filepath: str, **kwargs) -> StepLogger:
    """Create a StepLogger with configuration options"""
    return StepLogger(filepath, **kwargs)


# Usage example
if __name__ == "__main__":
    # Test the enhanced step logger
    logger = create_step_logger(
        "logs/test_reasoning.jsonl",
        log_level="DEBUG",
        enable_console=True,
        max_entries=1000
    )

    # Test standard logging methods
    logger.info("Starting test")
    logger.debug("Debug message", extra={"test": "data"})
    logger.warning("Warning message")

    # Test step logging
    logger.log_step(
        context="What is machine learning?",
        output="Machine learning is a subset of artificial intelligence...",
        metadata={"step": 1, "confidence": 0.85}
    )

    # Test error logging
    logger.log_error(
        error_message="Test error",
        context="During testing",
        metadata={"error_type": "test"}
    )

    # Test performance logging
    logger.log_performance(
        operation="reasoning_step",
        duration_ms=1500,
        success=True,
        metadata={"tokens": 250}
    )

    # Get statistics
    stats = logger.get_stats()
    print("Logger Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Search for entries
    reasoning_entries = logger.search_entries(entry_type="reasoning_step")
    print(f"\nFound {len(reasoning_entries)} reasoning step entries")

    print("Enhanced StepLogger test completed!")
