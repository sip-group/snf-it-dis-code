#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "This bash script downloads the datasets required for predictor patent on Gallager server."
   echo "It will only download cropped versions stored in rcod/ folder."
   echo "For each dataset type, it will download original and fakes only based on printer HPI55."
   echo "/!\ You should have a private/public key setup with Gallager to run this script."
   echo "Last test: 15.11.2022"
   echo
   echo "Syntax: download_dataset [-h|p|c|v]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Path to target folder."
   echo "u     User id to connect to Gallager server"
   echo "r     String of runs you would like to download : \"1 2\" or \"2 4 6\" ... "
   echo "e     Also download Unet estimates"
   echo "c     only check whether the paths are still valid (not implemented)."
   echo "s     Asks the script to run silently."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

# Set variables
target=""
user_id=""
runs="1"
only_check=False
verbose=True
dl_unet=False

############################################################
# Process the input options. Add options as needed.        #
############################################################
# Get the options
while getopts "ht:u:r:ecs" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      t) # Enter path to target
         target=$OPTARG;;
      u) # Enter your userid
         user_id=$OPTARG;;
      r) # Enter your userid
         runs=$OPTARG;;
      e) # Download Unet estimates
         dl_unet=True;;
      c) # Only check if links are still valid
         only_check=True;;
      s) # Verbose mode
         verbose=False;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

############################################################
# Source paths                                             #
############################################################

server='gallager.unige.ch'
main_folder="/ndata/chaban/cdp/images/d1_ps1_dens0.5_rep1"

############################################################
# Create directories                                       #
############################################################

if [ "$verbose" == True ]
then
  echo "Creating folders in $target"
fi

mkdir "$target/templates" "$target/scanner" "$target/iphone" "$target/samsung"
mkdir "$target/scanner/original" "$target/scanner/fake"

if [ "$dl_unet" == True ]
then
  mkdir "$target/scanner/template_original_unet"
  mkdir "$target/scanner/template_fake_unet"
fi

for run in $runs
do
  mkdir "$target/iphone/run_$run"
  mkdir "$target/samsung/run_$run"

  mkdir "$target/iphone/run_$run/original" "$target/iphone/run_$run/fake"
  mkdir "$target/samsung/run_$run/original" "$target/samsung/run_$run/fake"

  if [ "$dl_unet" == True ]
  then
    mkdir "$target/iphone/run_$run/template_original_unet" "$target/samsung/run_$run/template_original_unet"
    mkdir "$target/iphone/run_$run/template_fake_unet" "$target/samsung/run_$run/template_fake_unet"
  fi

done


############################################################
# Download files                                           #
############################################################

# ----------------------------------------------------------
# Download templates ---------------------------------------
# ----------------------------------------------------------

if [ "$verbose" == True ]
then
  echo "Downloading files in templates"
fi

templates="orig_template/rcod"

scp "$user_id"@"$server":"$main_folder/$templates/*" "$target/templates" > /dev/null

# ----------------------------------------------------------
# Download scanner -----------------------------------------
# ----------------------------------------------------------

if [ "$verbose" == True ]
then
  echo "Downloading files in scanner"
fi

orig_scanner="orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun1_scandpi2400/rcod"
fake_scanner="fake_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/scanrun1_scandpi2400/rcod"
orig_unet_scanner="orig_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG/scanrun1_scandpi2400/estimation"
fake_unet_scanner="fake_scan/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/scanrun1_scandpi2400/estimation"

scp "$user_id"@"$server":"$main_folder/$orig_scanner/*" "$target/scanner/original" > /dev/null
scp "$user_id"@"$server":"$main_folder/$fake_scanner/*" "$target/scanner/fake" > /dev/null

if [ "$dl_unet" == True ]
then
  scp "$user_id"@"$server":"$main_folder/$orig_unet_scanner/*" "$target/scanner/template_original_unet" > /dev/null
  scp "$user_id"@"$server":"$main_folder/$fake_unet_scanner/*" "$target/scanner/template_fake_unet" > /dev/null
fi

# ----------------------------------------------------------
# Download iphone ------------------------------------------
# ----------------------------------------------------------

if [ "$verbose" == True ]
then
  echo "Downloading files in iphone"
fi

for run in $runs
do

  orig_iphone="orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/iPhone12Pro_run${run}_ss100_focal12_apperture1/rcod"
  fake_iphone="fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/iPhone12Pro_run${run}_ss100_focal12_apperture1/rcod"
  orig_unet_iphone="orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/iPhone12Pro_run${run}_ss100_focal12_apperture1/unet_estimation"
  fake_unet_iphone="fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/iPhone12Pro_run${run}_ss100_focal12_apperture1/unet_estimation"

  scp "$user_id"@"$server":"$main_folder/$orig_iphone/*" "$target/iphone/run_$run/original" > /dev/null
  scp "$user_id"@"$server":"$main_folder/$fake_iphone/*" "$target/iphone/run_$run/fake" > /dev/null

  if [ "$dl_unet" == True ]
  then
    scp "$user_id"@"$server":"$main_folder/$orig_unet_iphone/*" "$target/iphone/run_$run/template_original_unet" > /dev/null
    scp "$user_id"@"$server":"$main_folder/$fake_unet_iphone/*" "$target/iphone/run_$run/template_fake_unet" > /dev/null
  fi

done

# ----------------------------------------------------------
# Download samsung -----------------------------------------
# ----------------------------------------------------------

if [ "$verbose" == True ]
then
  echo "Downloading files in samsung"
fi

for run in $runs
do

  orig_samsung="orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/SamsungGN20U_run${run}_ss100_focal12_apperture1/rcod"
  fake_samsung="fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/SamsungGN20U_run${run}_ss100_focal12_apperture1/rcod"
  orig_unet_samsung="orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/SamsungGN20U_run${run}_ss100_focal12_apperture1/unet_estimation"
  fake_unet_samsung="fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55/SamsungGN20U_run${run}_ss100_focal12_apperture1/unet_estimation"

  scp "$user_id"@"$server":"$main_folder/$orig_samsung/*" "$target/samsung/run_$run/original" > /dev/null
  scp "$user_id"@"$server":"$main_folder/$fake_samsung/*" "$target/samsung/run_$run/fake" > /dev/null

  if [ "$dl_unet" == True ]
    then
      scp "$user_id"@"$server":"$main_folder/$orig_unet_samsung/*" "$target/samsung/run_$run/template_original_unet" > /dev/null
      scp "$user_id"@"$server":"$main_folder/$fake_unet_samsung/*" "$target/samsung/run_$run/template_fake_unet" > /dev/null
    fi

done

echo "done"
