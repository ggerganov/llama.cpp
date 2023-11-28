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



# Function to install Zenity
install_zenity() {
    echo "Try to install Zenity with $1..."
    if ! $1 install zenity -y; then
        echo "Error: Zenity could not be installed."
        exit 1
    fi
    echo "Zenity was successfully installed."
}

# Check whether Zenity is already installed
if ! command -v zenity &> /dev/null; then
    # Zenity is not installed, try to find the package manager
    PACKAGE_MANAGERS=(brew apt apt-get yum pacman)
    for manager in "${PACKAGE_MANAGERS[@]}"; do
        if command -v $manager &> /dev/null; then
            # Package manager found, ask the user for permission
            read -p "Zenity is not installed. Would you like to install Zenity with $manager? (y/N) " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                # User has agreed, install Zenity
                install_zenity $manager
                break
            else
                echo "Installation canceled."
                exit 1
            fi
        fi
    done
    if ! command -v zenity &> /dev/null; then
        echo "No supported package manager found or Zenity could not be installed. Please install Zenity manually."
        exit 1
    fi
fi



model_selection() {
  # User selects a file or folder
  model_path=$(zenity --file-selection --title="Select Model File or Folder" --filename="$HOME/" --file-filter="*.gguf" --file-filter="*" --width=300 --height=400)
  exit_status=$?

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
    zenity --error --title="Invalid Selection" --text="The selected path is not valid."
    return
  fi

  # Selection menu for models found
  model_choice=$(zenity --list --title="Select a Model File" --column="Index" --column="Model File" $(for i in "${!model_files[@]}"; do echo "$((i+1))" "$(basename "${model_files[$i]}")"; done) --width=300 --height=400)
  exit_status=$?

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Set path to the selected model
  model_path=${model_files[$((model_choice-1))]}
}



multimodal_model_selection() {
  # User selects a file or folder
  mmproj_path=$(zenity --file-selection --title="Select Multimodal Model File or Folder" --filename="$HOME/" --file-filter="*.bin" --file-filter="*" --width=300 --height=400)
  exit_status=$?

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
    zenity --error --title="Invalid Selection" --text="The selected path is not valid."
    return
  fi

  # Selection menu for models found
  multi_modal_choice=$(zenity --list --title="Select a Multimodal Model File" --column="Index" --column="Model File" $(for i in "${!multi_modal_files[@]}"; do echo "$((i+1))" "$(basename "${multi_modal_files[$i]}")"; done) --width=300 --height=400)
  exit_status=$?

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Set path to the selected model
  mmproj_path=${multi_modal_files[$((multi_modal_choice-1))]}
}



options() {
  # Show form for entering the options
  form_values=$(zenity --forms --title="Set Options" --text="Enter the values for the following options:" --add-entry="Number of Threads (-t):" --add-entry="Context Size (-c):" --add-entry="Batch Size (-b):" --add-entry="GPU Layers (-ngl):" --separator="|" --width=300 --height=400)
  exit_status=$?

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Save the entered values in the corresponding variables
  IFS="|" read -r threads ctx_size batch_size n_gpu_layers <<< "$form_values"
}



further_options() {
  # Initial values for the checkboxes based on current settings
  cb_value=$([ "$cont_batching" = "on" ] && echo "TRUE" || echo "FALSE")
  mlock_value=$([ "$mlock" = "on" ] && echo "TRUE" || echo "FALSE")
  no_mmap_value=$([ "$no_mmap" = "on" ] && echo "TRUE" || echo "FALSE")

  # Show dialog for setting options
  choices=$(zenity --list --title="Boolean Options" --text="Select options:" --checklist --column="Select" --column="Option" TRUE "Continuous Batching (-cb)" FALSE "Memory Lock (--mlock)" FALSE "No Memory Map (--no-mmap)" --width=300 --height=400)
  exit_status=$?

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
      "Continuous Batching (-cb)") cont_batching="on" ;;
      "Memory Lock (--mlock)") mlock="on" ;;
      "No Memory Map (--no-mmap)") no_mmap="on" ;;
    esac
  done
}



advanced_options() {
  # Input fields for Advanced Options
  advanced_values=$(zenity --forms --title="Advanced Server Configuration" --text="Enter the advanced configuration options:" --add-entry="Host IP:" --add-entry="Port:" --add-entry="Additional Options:" --separator="|" --width=300 --height=400)
  exit_status=$?

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Read the entries and save them in the corresponding variables
  IFS="|" read -r host port advanced_options <<< "$advanced_values"
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
  read -p 'Press Enter to continue...'
}



# Function to save the current configuration
save_config() {
  config_file=$(zenity --file-selection --title="Save Configuration File" --filename="$HOME/" --width=300 --height=400)
  exit_status=$?

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

  zenity --info --title="Configuration Saved" --text="Configuration has been saved to $config_file" --width=300 --height=400
}



# Function for loading the configuration from a file
load_config() {
  config_file=$(zenity --file-selection --title="Load Configuration File" --filename="$HOME/" --width=300 --height=400)
  exit_status=$?

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

  # Check whether the configuration file exists
  if [ ! -f "$config_file" ]; then
    zenity --error --title="File Not Found" --text="The file $config_file was not found." --width=300 --height=400
    return
  fi

  # Load configuration from the file
  source "$config_file"

  zenity --info --title="Configuration Loaded" --text="Configuration has been loaded from $config_file" --width=300 --height=400
}



# Function to show the main menu
show_main_menu() {
  while true; do
    selection=$(zenity --list --title="Main Menu" --text="Please select:" --cancel-label="Exit" --column="Index" --column="Option" 1 "Model Selection" 2 "Multimodal Model Selection" 3 "Options" 4 "Further Options" 5 "Advanced Options" 6 "Save Config" 7 "Load Config" 8 "Start Server" --width=300 --height=400)
    exit_status=$?

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
