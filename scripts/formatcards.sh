#!/bin/bash

# Find all 512GB SD cards
sd_cards=$(lsblk -d -o name,size | grep "476.7G" | awk '{print $1}')

# Format each SD card
for sd_card in $sd_cards; do
    echo "Formatting $sd_card..."
    sudo umount /dev/${sd_card}1
    sudo mkfs.exfat /dev/${sd_card}1  # Change mkfs command as per your desired file system
    sudo sync
    sudo eject /dev/${sd_card}
done
