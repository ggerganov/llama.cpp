#!/bin/bash

# Set default values
model_path="./models/"
mmproj_path=""
threads=4
ctx_size=512
batch_size=512
n_gpu_layers=0
cont_batching="off"
mlock="off"
no_mmap="off"
host="127.0.0.1"
port="8080"
advanced_options=""



model_selection() {
    # User selects a file or folder
  exec 3>&1
  model_path=$(dialog --backtitle "Model Selection" \
                      --title "Select Model File or Folder" \
                      --fselect "$HOME/" 14 60 \
                      2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # If a folder has been selected, search for *.gguf files
  if [ -d "$model_path" ]; then
    model_files=($(find "$model_path" -name "*.gguf" 2>/dev/null))
  elif [ -f "$model_path" ]; then
    model_files=("$model_path")
  else
    dialog --backtitle "Model Selection" \
           --title "Invalid Selection" \
           --msgbox "The selected path is not valid." 7 50
    return
  fi

# Selection menu for models found
exec 3>&1
model_choice=$(dialog --backtitle "Model Selection" \
                      --title "Select a Model File" \
                      --menu "Choose one of the found models:" 15 60 4 \
                      $(for i in "${!model_files[@]}"; do echo "$((i+1))" "$(basename "${model_files[$i]}")"; done) \
                      2>&1 1>&3)
exit_status=$?
exec 3>&-

# Check whether user has selected 'Cancel'
if [ $exit_status = 1 ]; then
  return
fi

# Set path to the selected model
model_path=${model_files[$((model_choice-1))]}
}



multimodal_model_selection() {
    # User selects a file or folder
  exec 3>&1
  mmproj_path=$(dialog --backtitle "Multimodal Model" \
                      --title "Select Model File or Folder" \
                      --fselect "$HOME/" 14 60 \
                      2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # If a folder has been selected, search for *.bin files
  if [ -d "$mmproj_path" ]; then
    multi_modal_files=($(find "$mmproj_path" -name "*.bin" 2>/dev/null))
  elif [ -f "$mmproj_path" ]; then
    multi_modal_files=("$mmproj_path")
  else
    dialog --backtitle "Multimodal Model" \
           --title "Invalid Selection" \
           --msgbox "The selected path is not valid." 7 50
    return
  fi

# Selection menu for models found
exec 3>&1
multi_modal_choice=$(dialog --backtitle "Multimodal Model" \
                      --title "Select a Model File" \
                      --menu "Choose one of the found models:" 15 60 4 \
                      $(for i in "${!multi_modal_files[@]}"; do echo "$((i+1))" "$(basename "${multi_modal_files[$i]}")"; done) \
                      2>&1 1>&3)
exit_status=$?
exec 3>&-

# Check whether user has selected 'Cancel'
if [ $exit_status = 1 ]; then
  return
fi

# Set path to the selected model
mmproj_path=${multi_modal_files[$((multi_modal_choice-1))]}
}



options() {
  # Show form for entering the options
  exec 3>&1
  form_values=$(dialog --backtitle "Options Configuration" \
                       --title "Set Options" \
                       --form "Enter the values for the following options:" \
                       15 50 0 \
                       "Number of Threads (-t):" 1 1 "$threads" 1 25 25 5 \
                       "Context Size (-c):" 2 1 "$ctx_size" 2 25 25 5 \
                       "Batch Size (-b):" 3 1 "$batch_size" 3 25 25 5 \
                       "GPU Layers (-ngl):" 4 1 "$n_gpu_layers" 4 25 25 5 \
                       2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Save the entered values in the corresponding variables
  IFS=$'\n' read -r threads ctx_size batch_size n_gpu_layers <<< "$form_values"
}



further_options() {
  # Initial values for the checkboxes based on current settings
  cb_value=$([ "$cont_batching" = "on" ] && echo "on" || echo "off")
  mlock_value=$([ "$mlock" = "on" ] && echo "on" || echo "off")
  no_mmap_value=$([ "$no_mmap" = "on" ] && echo "on" || echo "off")

  # Show dialog for setting options
  exec 3>&1
  choices=$(dialog --backtitle "Further Options" \
                   --title "Boolean Options" \
                   --checklist "Select options:" 15 60 3 \
                   "1" "Continuous Batching (-cb)" $cb_value \
                   "2" "Memory Lock (--mlock)" $mlock_value \
                   "3" "No Memory Map (--no-mmap)" $no_mmap_value \
                   2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Set options based on user selection
  cont_batching="off"
  mlock="off"
  no_mmap="off"
  for choice in $choices; do
    case $choice in
      1) cont_batching="on" ;;
      2) mlock="on" ;;
      3) no_mmap="on" ;;
    esac
  done
}



advanced_options() {
  # Input fields for Advanced Options
  exec 3>&1
  advanced_values=$(dialog --backtitle "Advanced Options" \
                           --title "Advanced Server Configuration" \
                           --form "Enter the advanced configuration options:" \
                           15 60 0 \
                           "Host IP:" 1 1 "$host" 1 15 15 0 \
                           "Port:" 2 1 "$port" 2 15 5 0 \
                           "Additional Options:" 3 1 "$advanced_options" 3 15 30 0 \
                           2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Read the entries and save them in the corresponding variables
  read -r host port advanced_options <<< "$advanced_values"
}



start_server() {
  # Compiling the command with the selected options
  cmd="./server"
  [ -n "$model_path" ] && cmd+=" -m $model_path"
  [ -n "$mmproj_path" ] && cmd+=" --mmproj $mmproj_path"
  [ "$threads" -ne 4 ] && cmd+=" -t $threads"
  [ "$ctx_size" -ne 512 ] && cmd+=" -c $ctx_size"
  [ "$batch_size" -ne 512 ] && cmd+=" -b $batch_size"
  [ "$n_gpu_layers" -ne 0 ] && cmd+=" -ngl $n_gpu_layers"
  [ "$cont_batching" = "on" ] && cmd+=" -cb"
  [ "$mlock" = "on" ] && cmd+=" --mlock"
  [ "$no_mmap" = "off" ] && cmd+=" --no-mmap"
  [ -n "$host" ] && cmd+=" --host $host"
  [ -n "$port" ] && cmd+=" --port $port"
  [ -n "$advanced_options" ] && cmd+=" $advanced_options"

    eval "$cmd"
  read -p 'Do not forget to quit the server later with Ctrl+C as soon as you are finished. Press Enter to continue...'
}



# Function to save the current configuration
save_config() {
  exec 3>&1
  config_file=$(dialog --backtitle "Save Configuration" \
                       --title "Save Configuration File" \
                       --fselect "$HOME/" 14 60 \
                       2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Saving the configuration to the file
  cat > "$config_file" << EOF
model_path=$model_path
mmproj_path=$mmproj_path
threads=$threads
ctx_size=$ctx_size
batch_size=$batch_size
n_gpu_layers=$n_gpu_layers
cont_batching=$cont_batching
mlock=$mlock
no_mmap=$no_mmap
host=$host
port=$port
advanced_options=$advanced_options
EOF

  dialog --backtitle "Save Configuration" \
         --title "Configuration Saved" \
         --msgbox "Configuration has been saved to $config_file" 7 50
}



# Function for loading the configuration from a file
load_config() {
  exec 3>&1
  config_file=$(dialog --backtitle "Load Configuration" \
                       --title "Load Configuration File" \
                       --fselect "$HOME/" 14 60 \
                       2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Check whether the configuration file exists
  if [ ! -f "$config_file" ]; then
    dialog --backtitle "Load Configuration" \
           --title "File Not Found" \
           --msgbox "The file $config_file was not found." 7 50
    return
  fi

  # Load configuration from the file
  source "$config_file"

  dialog --backtitle "Load Configuration" \
         --title "Configuration Loaded" \
         --msgbox "Configuration has been loaded from $config_file" 7 50
}



# Function to show the main menu
show_main_menu() {
  while true; do
    exec 3>&1
    selection=$(dialog \
      --backtitle "Server Configuration" \
      --title "Main Menu" \
      --clear \
      --cancel-label "Exit" \
      --menu "Please select:" 15 50 6 \
      "1" "Model Selection" \
      "2" "Multimodal Model Selection" \
      "3" "Options" \
      "4" "Further Options" \
      "5" "Advanced Options" \
      "6" "Save Config" \
      "7" "Load Config" \
      "8" "Start Server" \
      2>&1 1>&3)
    exit_status=$?
    exec 3>&-

    # Check whether user has selected 'Exit'
    if [ $exit_status = 1 ]; then
      clear
      exit
    fi

    # Call up the corresponding function based on the selection
    case $selection in
      1) model_selection ;;
      2) multimodal_model_selection ;;
      3) options ;;
      4) further_options ;;
      5) advanced_options ;;
      6) save_config ;;
      7) load_config ;;
      8) start_server ;;
      *) clear ;;
    esac
  done
}



# Show main menu
show_main_menu
