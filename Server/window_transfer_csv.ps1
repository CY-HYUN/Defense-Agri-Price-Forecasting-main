$sourceDirectory = "C:\path\to\csv_files"   # Windows에서 CSV 파일이 있는 디렉터리
$destination = "<username@<Ubuntu_IP>:/home/username/path/to/csv_files"  # Ubuntu 디렉터리
$sshKey = "C:\Users\<username>\.ssh\id_rsa"  # SSH 개인 키 경로

# CSV 파일 목록 가져오기
$csvFiles = Get-ChildItem -Path $sourceDirectory -Filter *.csv

# 파일 전송
foreach ($file in $csvFiles) {
    $filePath = $file.FullName
    Write-Host "Uploading $filePath to $destination"
	
    scp -i $sshKey $filePath $destination
    if ($?) {
        Write-Host "Successfully uploaded $filePath"
		
        # 파일 이동 또는 삭제
        Move-Item $filePath "$sourceDirectory\$($file.Name)" -Force
    } else {
        Write-Host "Failed to upload $filePath"
