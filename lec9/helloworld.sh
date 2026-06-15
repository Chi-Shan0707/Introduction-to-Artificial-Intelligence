

#!/bin/bash


echo "input folder A $folder_a"

echo "input folder B $folder_b"

mkdir -p "folder_b"

for ext in jpg png py txt
do
    count=$(ls"$folder_a*" | grep -c "\.$ext$" )
    echo "number of $ext files in folder A is $count"

    # echo "number of $ext files in folder A is $count"


    if [ $count -gt 1 ]; then
        cp "$folder_a"/*.$ext"$folder_b"
    fi
done


