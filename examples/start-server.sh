#!/bin/bash

# Determine the size of the terminal window
TERMINAL_HEIGHT=$(tput lines)
TERMINAL_WIDTH=$(tput cols)

# Calculate a size for the dialog box as a percentage of the terminal to make sure it fits
DIALOG_HEIGHT=$((TERMINAL_HEIGHT * 5 / 8)) # approx. golden ratio
DIALOG_WIDTH=$((TERMINAL_WIDTH * 5 / 8)) # approx. golden ratio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Set default values
model_path="$SCRIPT_DIR/../models/"
mmproj_path="$SCRIPT_DIR/../models/"
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



# Get absolute path of a file or directory
get_absolute_path() {
  local target_file=$1

  if command -v readlink &>/dev/null; then
    echo "$(readlink -f "$target_file")"
  elif command -v greadlink &>/dev/null; then
    echo "$(greadlink -f "$target_file")"
  else
    echo "Error: Neither readlink nor greadlink is available."
    exit 1
  fi
}



# Install Dialog if missing
install_dialog() {
    echo "Try to install Dialog with $1..."
    if ! $1 install dialog; then
        echo "Error: Dialog could not be installed."
        exit 1
    fi
    echo "Dialog was successfully installed."
}

# Check whether Dialog is already installed
if ! command -v dialog &> /dev/null; then
    # Dialog is not installed, try to find the package manager. I start with brew since this is the only cross-platform pkg-manager.
    PACKAGE_MANAGERS=(brew apt apt-get yum pacman)
    for manager in "${PACKAGE_MANAGERS[@]}"; do
        if command -v $manager &> /dev/null; then
            # If package manager found, ask user for permission
            read -p "Dialog is not installed. Would you like to install Dialog with $manager? (y/N) " response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                # If user has agreed, install Dialog
                install_dialog $manager
                break
            else
                echo "Installation canceled."
                exit 1
            fi
        fi
    done
    if ! command -v dialog &> /dev/null; then
        echo "No supported package manager found or Dialog could not be installed. Please install Dialog manually."
        exit 1
    fi
fi



model_selection_warning() {
  dialog --title "Note" --msgbox "\n\n\nPlease note!\n\nTo navigate to a folder, please press the space bar twice. To return to a higher-level folder, press the Backspace key.\n\n\nAlternatively, you can also enter the desired path manually in the lower path field. \n\n\nOnly confirm your selection with the Enter key once you have selected the file – or the desired folder to be searched recursively." $DIALOG_HEIGHT $DIALOG_WIDTH
}



model_selection() {
  # User selects a file or folder
  exec 3>&1

  # Set initial directory for the file selection dialog
  INITIAL_DIR="$SCRIPT_DIR/../models/"

  model_path=$(dialog --backtitle "Model Selection" \
                      --title "Select Model File or Folder" \
                      --fselect "$INITIAL_DIR" $DIALOG_HEIGHT $DIALOG_WIDTH \
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
    # Check whether files have been found
    if [ ${#model_files[@]} -eq 0 ]; then
      dialog --backtitle "Model Selection" \
             --title "No Models Found" \
             --msgbox "\n\n\nNo model files (*.gguf) were found in the selected directory." $DIALOG_HEIGHT $DIALOG_WIDTH
      return
    fi
  elif [ -f "$model_path" ]; then
    model_files=("$model_path")
  else
    dialog --backtitle "Model Selection" \
           --title "Invalid Selection" \
           --msgbox "\n\n\nThe selected path is not valid." $DIALOG_HEIGHT $DIALOG_WIDTH
    return
  fi

# Selection menu for models found
exec 3>&1
model_choice=$(dialog --backtitle "Model Selection" \
                      --title "Select a Model File" \
                      --menu "Choose one of the found models:" $DIALOG_HEIGHT $DIALOG_WIDTH \
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
  INITIAL_DIR="$SCRIPT_DIR/../models/"

  mmproj_path=$(dialog --backtitle "Multimodal Model Selection" \
                       --title "Select Multimodal Model File or Folder" \
                       --fselect "$INITIAL_DIR"  $DIALOG_HEIGHT $DIALOG_WIDTH \
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
  # Check whether files have been found
  if [ ${#multi_modal_files[@]} -eq 0 ]; then
    dialog --backtitle "Multimodal Model Selection" \
           --title "No Multimodal Models Found" \
           --msgbox "\n\n\nNo multimodal model files (*.bin) were found in the selected directory." $DIALOG_HEIGHT $DIALOG_WIDTH
    return
  fi
elif [ -f "$mmproj_path" ]; then
  multi_modal_files=("$mmproj_path")
else
  dialog --backtitle "Multimodal Model Selection" \
         --title "Invalid Selection" \
         --msgbox "\n\n\nThe selected path is not valid." $DIALOG_HEIGHT $DIALOG_WIDTH
  return
fi

# Selection menu for models found
exec 3>&1
multi_modal_choice=$(dialog --backtitle "Multimodal Model" \
                            --title "Select a Model File" \
                            --menu "Choose one of the found models:" $DIALOG_HEIGHT $DIALOG_WIDTH 8 \
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
                       $DIALOG_HEIGHT $DIALOG_WIDTH 0 \
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
                   --checklist "Select options:"  $DIALOG_HEIGHT $DIALOG_WIDTH 3 \
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
                            $DIALOG_HEIGHT $DIALOG_WIDTH 0 \
                           "Host IP:" 1 1 "$host" 1 25 15 0 \
                           "Port:" 2 1 "$port" 2 25 5 0 \
                           "Additional Options:" 3 1 "$advanced_options" 3 25 30 0 \
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



# Function to save the current configuration
save_config() {
  exec 3>&1
  config_file=$(dialog --backtitle "Save Configuration" \
                       --title "Save Configuration File" \
                       --fselect "$SCRIPT_DIR/" $DIALOG_HEIGHT $DIALOG_WIDTH \
                       2>&1 1>&3)
  exit_status=$?
  exec 3>&-

  # Check whether user has selected 'Cancel'
  if [ $exit_status = 1 ]; then
    return
  fi

absolute_model_path=$(get_absolute_path "$model_path")
absolute_mmproj_path=$(get_absolute_path "$mmproj_path")

# Saving the configuration to the file with absolute paths using custom function
cat > "$config_file" << EOF
model_path=$absolute_model_path
mmproj_path=$absolute_mmproj_path
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
         --msgbox "\n\n\nYour configuration has been saved to\n\n$config_file" $DIALOG_HEIGHT $DIALOG_WIDTH
}



# loading the configuration from a file
load_config() {
  exec 3>&1
  config_file=$(dialog --backtitle "Load Configuration" \
                       --title "Load Configuration File" \
                       --fselect "$SCRIPT_DIR/" $DIALOG_HEIGHT $DIALOG_WIDTH \
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
           --msgbox "\n\n\nThe file $config_file was not found." $DIALOG_HEIGHT $DIALOG_WIDTH
    return
  fi

  # Load configuration from the file
  source "$config_file"

  # Convert model paths to absolute paths
  model_path=$(get_absolute_path "$model_path")
  mmproj_path=$(get_absolute_path "$mmproj_path")

  dialog --backtitle "Load Configuration" \
         --title "Configuration Loaded" \
         --msgbox "\n\n\nConfiguration has been loaded successfully from\n\n$config_file" $DIALOG_HEIGHT $DIALOG_WIDTH
}



# Checking the existence of the server executable
check_server_executable() {
  local server_executable="$SCRIPT_DIR/../server"
  local makefile_path="$SCRIPT_DIR/../Makefile"
  local cmake_lists_path="$SCRIPT_DIR/../CMakeLists.txt"

  if [ ! -f "$server_executable" ]; then
    # Server executable does not exist, check Makefile and CMakeLists.txt
    if [ -f "$makefile_path" ] && [ -f "$cmake_lists_path" ]; then
      # Offer the user to build the server now
      exec 3>&1
      response=$(dialog --title "Server Executable Missing" \
                        --yesno "\n\n\nThe server executable does not exist. Would you like to run 'make' to build it? Note: This will build a basic server without GPU acceleration. Please read the documentation if you need more options and run the build process manually." $DIALOG_HEIGHT $DIALOG_WIDTH \
                        2>&1 1>&3)
      exit_status=$?
      exec 3>&-

      if [ $exit_status = 0 ]; then
        # User has agreed, run 'make'
        (cd "$SCRIPT_DIR/.." && make)
        # Check if 'make' was successful
        if [ ! -f "$server_executable" ]; then
          dialog --title "Build Failed" --msgbox "\n\n\nThe server could not be built. Please check the build process manually." $DIALOG_HEIGHT $DIALOG_WIDTH
          exit 1
        fi
      else
        # User has rejected or pressed ESC
        dialog --title "Build Canceled" --msgbox "\n\n\nServer build was canceled. Cannot start the server without the executable." $DIALOG_HEIGHT $DIALOG_WIDTH
        exit 1
      fi
    else
      # Makefile and CMakeLists.txt do not exist
      dialog --title "Critical Error" --msgbox "\n\n\nMakefile and CMakeLists.txt are missing. This script may not be in the correct directory. Please make sure this script is in its correct directory." $DIALOG_HEIGHT $DIALOG_WIDTH
      exit 1
    fi
  fi
}



confirm_and_start_server() {
  # Check whether model_path refers to a valid .gguf file
  if [[ ! "$model_path" =~ \.gguf$ ]] || [ ! -f "$model_path" ]; then
    dialog --title "Invalid Model File" --msgbox "\n\n\nThe selected model file ($model_path) is not valid or does not end with .gguf or a model file was not selected yet.\n\n\nPlease select a valid .gguf model file." $DIALOG_HEIGHT $DIALOG_WIDTH
    return 1
  fi

  # Show the compiled command in a dialog box
  dialog --title "Server Start Confirmation" --yesno "\n\n\nThe server will be started with the following command:\n\n$cmd\n\nDo not forget to close the server with Ctrl+C as soon as you are finished.\n\nWould you like to continue?" $DIALOG_HEIGHT $DIALOG_WIDTH

  # Check exit status of dialog
  response=$?
  case $response in
    0) eval "$cmd" ;;  # User has selected 'Yes', execute the server command
    1) return 1 ;;     # User has selected 'No', return to main menu
    255) echo "[ESC] key pressed.";;  # The user has pressed ESC
  esac
}



start_server() {
  # Absolute path to the server executable
  SERVER_CMD="$SCRIPT_DIR/../server"

  # Compiling the command with the selected options
  cmd="$SERVER_CMD"
  [ -n "$model_path" ] && cmd+=" -m $model_path"
  [ -n "$mmproj_path" ] && cmd+=" --mmproj $mmproj_path"
  [ "$threads" -ne 4 ] && cmd+=" -t $threads"
  [ "$ctx_size" -ne 512 ] && cmd+=" -c $ctx_size"
  [ "$batch_size" -ne 512 ] && cmd+=" -b $batch_size"
  [ "$n_gpu_layers" -ne 0 ] && cmd+=" -ngl $n_gpu_layers"
  [ "$cont_batching" = "on" ] && cmd+=" -cb"
  [ "$mlock" = "on" ] && cmd+=" --mlock"
  [ "$no_mmap" = "on" ] && cmd+=" --no-mmap"
  [ -n "$host" ] && cmd+=" --host $host"
  [ -n "$port" ] && cmd+=" --port $port"
  [ -n "$advanced_options" ] && cmd+=" $advanced_options"

  confirm_and_start_server || return
  }



# Function to confirm exit
confirm_exit() {
  exec 3>&1
  selection=$(dialog \
    --backtitle "Confirm Exit" \
    --title "Are you sure?" \
    --yesno "Are you sure you want to exit?" 7 60 \
    2>&1 1>&3)
  exit_status=$?
  exec 3>&-
  return $exit_status
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
      --menu "Welcome to llama.cpp Dialog" $DIALOG_HEIGHT $DIALOG_WIDTH 8 \
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

    # Check whether user has unintentionally selected 'Exit'
    if [ $exit_status = 1 ]; then
      confirm_exit
      if [ $? = 0 ]; then
        clear
        exit
      fi
    else

      # Call up the corresponding function based on the selection
      case $selection in
        1) model_selection_warning; model_selection ;;
        2) model_selection_warning; multimodal_model_selection ;;
        3) options ;;
        4) further_options ;;
        5) advanced_options ;;
        6) save_config ;;
        7) load_config ;;
        8) start_server ;;
        *) clear ;;
      esac
    fi
  done
}



# Show main menu
show_main_menu
