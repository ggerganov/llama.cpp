# Get the working directory
$modelsPath = $pwd

# Define the file with the list of hashes and filenames
$hashListPath = "SHA256SUMS"

# Check if the hash list file exists
if (-not(Test-Path -Path $hashListPath)) {
    Write-Error "Hash list file not found: $hashListPath"
    exit 1
}

# Read the hash file content and split it into an array of lines
$hashList = Get-Content -Path $hashListPath
$hashLines = $hashList -split "`n"

# Create an array to store the results
$results = @()

# Loop over each line in the hash list
foreach ($line in $hashLines) {

  # Split the line into hash and filename
  $hash, $filename = $line -split "  "

  # Get the full path of the file by joining the models path and the filename
  $filePath = Join-Path -Path $modelsPath -ChildPath $filename
  
  # Informing user of the progress of the integrity check
  Write-Host "Verifying the checksum of $filePath"
  
  # Check if the file exists
  if (Test-Path -Path $filePath) {

    # Calculate the SHA256 checksum of the file using certUtil
    $fileHash = certUtil -hashfile $filePath SHA256 | Select-Object -Index 1

    # Remove any spaces from the hash output
    $fileHash = $fileHash -replace " ", ""

    # Compare the file hash with the expected hash
    if ($fileHash -eq $hash) {
      $validChecksum = "V"
      $fileMissing = ""
    }
    else {
      $validChecksum = ""
      $fileMissing = ""
    }
  }
  else {
    $validChecksum = ""
    $fileMissing = "X"
  }

  # Add the results to the array
  $results += [PSCustomObject]@{
    "filename" = $filename
    "valid checksum" = $validChecksum
    "file missing" = $fileMissing
  }
}

# Output the results as a table
$results | Format-Table `
  -Property @{Expression={$_.filename}; Label="filename"; Alignment="left"}, `
            @{Expression={$_."valid checksum"}; Label="valid checksum"; Alignment="center"}, `
            @{Expression={$_."file missing"}; Label="file missing"; Alignment="center"} `
  -AutoSize
