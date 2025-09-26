#!/bin/bash

# 변수 설정
SOURCE_DIR="$HOME/path/to/csv"
BACKUP_DIR="$HOME/path/backup"
HDFS_DIR="/hdfs/path/csv"

# SOURCE_DIR에 파일이 있는지 확인
if [[ ! $(ls -1 "$SOURCE_DIR"/*.csv 2>/dev/null) ]]; then
    echo "$SOURCE_DIR에 처리할 CSV 파일이 없습니다."
    exit 0
fi

# 모든 CSV 파일 UTF-8로 인코딩 변경 (덮어쓰기) 및 HDFS 업로드
echo "$SOURCE_DIR 파일들 UTF-8로 인코딩 변환 후 업로드중..."
for file in "$SOURCE_DIR"/*.csv; do
    [[ -f "$file" ]] && iconv -f EUC-KR -t UTF-8 "$file" -o "$file.tmp" && mv "$file.tmp" "$file"

    CSV_DIR=$(basename "$file" .csv | sed -E 's/_20[0-9]{6}$//')
    CSV_DIR=${CSV_DIR^^}

    hdfs dfs -mkdir -p "$HDFS_DIR/$CSV_DIR"
    hdfs dfs -put -f "$file" "$HDFS_DIR/$CSV_DIR/"

done

echo "$BACKUP_DIR로 이동중..."
mv "$SOURCE_DIR"/*.csv "$BACKUP_DIR/"
