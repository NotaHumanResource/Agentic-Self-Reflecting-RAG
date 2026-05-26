#!/usr/bin/env python3
"""
Qdrant Vector Database Backup Manager
Handles backup, restore, and management of Qdrant Docker volumes
"""

import os
import sys
import subprocess
import argparse
import json
import tarfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qdrant_backup.log', encoding='utf-8')
    ]
)

class QdrantBackupManager:
    def __init__(self, backup_dir="D:/AI_DB_BACKUPS"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.volume_name = "qdrant_storage"
        self.container_name = "qdrant_server"
        
    def check_docker(self):
        """Check if Docker is running and accessible"""
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logging.info(f"[OK] Docker is running (version: {result.stdout.strip()})")
                return True
            else:
                logging.error(f"[ERROR] Docker check failed: {result.stderr}")
                return False
        except Exception as e:
            logging.error(f"[ERROR] Docker not accessible: {e}")
            return False
    
    def check_volume_exists(self):
        """Check if the Qdrant volume exists"""
        try:
            result = subprocess.run(
                ["docker", "volume", "inspect", self.volume_name],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logging.info(f"[OK] Volume '{self.volume_name}' exists")
                return True
            else:
                logging.error(f"[ERROR] Volume '{self.volume_name}' not found")
                return False
        except Exception as e:
            logging.error(f"[ERROR] Error checking volume: {e}")
            return False
    
    def create_backup(self, include_sql=False):
        """Create a backup of the Qdrant volume and optionally SQL database"""
        if not self.check_docker():
            return False
        
        if not self.check_volume_exists():
            return False
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = f"qdrant_backup_{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_filename
        
        logging.info(f"[BACKUP] Creating backup: {backup_filename}")
        logging.info(f"   Source: Docker volume '{self.volume_name}'")
        logging.info(f"   Destination: {backup_path}")
        
        try:
            # Use Docker to create backup
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.volume_name}:/source",
                "-v", f"{self.backup_dir}:/backup",
                "alpine",
                "tar", "czf", f"/backup/{backup_filename}", "-C", "/source", "."
            ]
            
            logging.info(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            
            if result.returncode == 0:
                # Check if backup file was created and get size
                if backup_path.exists():
                    file_size_mb = backup_path.stat().st_size / (1024 * 1024)
                    logging.info(f"[SUCCESS] Vector database backup completed!")
                    logging.info(f"   File: {backup_filename}")
                    logging.info(f"   Size: {file_size_mb:.2f} MB")
                    logging.info(f"   Location: {backup_path}")
                    
                    # Also backup SQL database if requested
                    if include_sql:
                        sql_backup = self.backup_sql_database()
                        if sql_backup:
                            logging.info(f"[SUCCESS] SQL database backup also completed!")
                            return {'vector': str(backup_path), 'sql': sql_backup}
                        else:
                            logging.warning("[WARNING] Vector backup succeeded but SQL backup failed")
                            return {'vector': str(backup_path), 'sql': False}
                    
                    return str(backup_path)
                else:
                    logging.error(f"[ERROR] Backup file not created: {backup_path}")
                    return False
            else:
                logging.error(f"[ERROR] Backup failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error("[ERROR] Backup timed out after 5 minutes")
            return False
        except Exception as e:
            logging.error(f"[ERROR] Backup error: {e}")
            return False
        
    def backup_all(self):
        """Create backups of both vector and SQL databases"""
        logging.info("[BACKUP-ALL] Starting complete backup of both databases...")
        
        result = self.create_backup(include_sql=True)
        
        if isinstance(result, dict):
            # Both backups attempted
            if result['vector'] and result['sql']:
                logging.info("[SUCCESS] Complete backup finished - both databases backed up!")
                return True
            elif result['vector']:
                logging.warning("[PARTIAL] Vector database backed up, but SQL backup failed")
                return True
            else:
                logging.error("[FAILED] Both backups failed")
                return False
        elif result:
            # Only vector backup (shouldn't happen with include_sql=True, but just in case)
            logging.warning("[PARTIAL] Only vector database was backed up")
            return True
        else:
            logging.error("[FAILED] Backup failed")
            return False
    
    def list_backups(self):
        """List all available backups"""
        logging.info("[LIST] Available Qdrant Backups:")
        
        backup_pattern = "qdrant_backup_*.tar.gz"
        backups = list(self.backup_dir.glob(backup_pattern))
        
        if not backups:
            logging.info(f"   No backups found in {self.backup_dir}")
            return []
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        backup_info = []
        for backup in backups:
            stat = backup.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            age_days = (datetime.now() - mod_time).days
            
            info = {
                'filename': backup.name,
                'path': str(backup),
                'size_mb': size_mb,
                'created': mod_time,
                'age_days': age_days
            }
            backup_info.append(info)
            
            logging.info(f"  [BACKUP] {backup.name}")
            logging.info(f"     Size: {size_mb:.2f} MB | Age: {age_days} days | Created: {mod_time}")
        
        return backup_info
    
    def restore_backup(self, backup_file, force=False):
        """Restore from a backup file"""
        backup_path = Path(backup_file)
        
        if not backup_path.exists():
            logging.error(f"[ERROR] Backup file not found: {backup_path}")
            return False
        
        if not self.check_docker():
            return False
        
        logging.info(f"[RESTORE] Starting Qdrant restore from: {backup_path.name}")
        
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            if result.stdout.strip() == self.container_name:
                if force:
                    logging.info("[WARNING] Stopping Qdrant container...")
                    subprocess.run(["docker", "stop", self.container_name], check=True)
                    subprocess.run(["docker", "rm", self.container_name], check=True)
                else:
                    logging.error("[ERROR] Qdrant container is running. Use --force to stop it automatically.")
                    return False
            
            # Remove existing volume
            logging.info("[RESTORE] Removing existing volume...")
            subprocess.run(["docker", "volume", "rm", self.volume_name], capture_output=True)
            
            # Create new volume
            logging.info("[RESTORE] Creating new volume...")
            subprocess.run(["docker", "volume", "create", self.volume_name], check=True)
            
            # Restore data
            logging.info("[RESTORE] Restoring data...")
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.volume_name}:/dest",
                "-v", f"{backup_path.parent}:/backup",
                "alpine",
                "tar", "xzf", f"/backup/{backup_path.name}", "-C", "/dest"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logging.info("[SUCCESS] Data restored successfully!")
                
                # Restart Qdrant container
                logging.info("[RESTORE] Starting Qdrant container...")
                start_cmd = [
                    "docker", "run", "-d", "--name", self.container_name,
                    "-p", "6333:6333", "-p", "6334:6334",
                    "-v", f"{self.volume_name}:/qdrant/storage",
                    "qdrant/qdrant:latest"
                ]
                
                subprocess.run(start_cmd, check=True)
                logging.info("[SUCCESS] Restore completed! Qdrant is running.")
                return True
            else:
                logging.error(f"[ERROR] Restore failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"[ERROR] Restore error: {e}")
            return False
        
    def backup_sql_database(self, db_path="LongTermMemory_data.db"):
        """Backup the SQL database using SQLite's backup API"""
        import sqlite3
        
        try:
            db_path = Path(db_path)
            if not db_path.exists():
                # Try relative path from script location
                db_path = Path(__file__).parent / db_path
            
            if not db_path.exists():
                logging.error(f"[ERROR] Database not found: {db_path}")
                return False
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"LongTermMemory_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            logging.info(f"[BACKUP] Creating SQL database backup: {backup_filename}")
            
            # Use SQLite's backup API for consistency
            source_conn = sqlite3.connect(str(db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            
            source_conn.backup(backup_conn)
            
            backup_conn.close()
            source_conn.close()
            
            file_size_mb = backup_path.stat().st_size / (1024 * 1024)
            logging.info(f"[SUCCESS] SQL backup completed: {file_size_mb:.2f} MB")
            return str(backup_path)
            
        except Exception as e:
            logging.error(f"[ERROR] SQL backup failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days=30):
        """Remove backups older than specified days"""
        logging.info(f"[CLEANUP] Cleaning backups older than {keep_days} days...")
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        backup_pattern = "qdrant_backup_*.tar.gz"
        backups = list(self.backup_dir.glob(backup_pattern))
        
        deleted_count = 0
        for backup in backups:
            mod_time = datetime.fromtimestamp(backup.stat().st_mtime)
            if mod_time < cutoff_date:
                logging.info(f"   Deleting: {backup.name}")
                backup.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logging.info(f"[SUCCESS] Cleanup completed. Removed {deleted_count} old backup(s).")
        else:
            logging.info("[SUCCESS] No old backups to clean.")
        
        return deleted_count

def main():
    parser = argparse.ArgumentParser(description="Qdrant Vector Database Backup Manager")
    parser.add_argument("--backup-dir", default="D:/AI_DB_BACKUPS", 
                       help="Backup directory (default: D:/AI_DB_BACKUPS)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a new vector database backup")
    
    # Backup-all command
    backup_all_parser = subparsers.add_parser("backup-all", help="Create backups of both vector and SQL databases")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all backups")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("backup_file", help="Path to backup file")
    restore_parser.add_argument("--force", action="store_true", 
                               help="Force stop Qdrant container if running")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean old backups")
    cleanup_parser.add_argument("--keep-days", type=int, default=30,
                               help="Keep backups newer than this many days (default: 30)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize backup manager
    manager = QdrantBackupManager(args.backup_dir)
    
    # Execute command
    if args.command == "backup":
        result = manager.create_backup()
        sys.exit(0 if result else 1)
        
    elif args.command == "backup-all":
        result = manager.backup_all()
        sys.exit(0 if result else 1)
        
    elif args.command == "list":
        manager.list_backups()
        
    elif args.command == "restore":
        result = manager.restore_backup(args.backup_file, args.force)
        sys.exit(0 if result else 1)
        
    elif args.command == "cleanup":
        manager.cleanup_old_backups(args.keep_days)

if __name__ == "__main__":
    main()