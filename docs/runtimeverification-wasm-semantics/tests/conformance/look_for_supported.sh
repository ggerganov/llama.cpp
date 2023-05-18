#!/bin/bash

# Execute the script from project top directory.

# This script considers all conformance tests currently listed as unparseable or unsupported, and tries parsing and executing them in turn.
# If both steps are successfull, the confromance test is deleted from the lists.
# If either step fails, the conformance test is added to the correct list and removed from the other.

unparsefile="tests/conformance/unparseable.txt"

unparseable=$(cat $unparsefile)
backends="llvm"

# Try all unparseable files, see if any new one parses and add them to the unsupported lists.
# Keep the files sorted and without duplicates so the diff is easy to follow.
for shortname in $unparseable $unsupported; do
    file=tests/wasm-tests/test/core/$shortname

    echo Trying to parse: $shortname
    echo ===============

    make $file.parse
    if [ $? -eq 0 ]
    then
        sed --in-place "/^$shortname\$/d" $unparsefile
        for backend in $backends; do
            unsuppfile="tests/conformance/unsupported-$backend.txt"
            echo $shortname >> $unsuppfile
            sort -u $unsuppfile -o $unsuppfile
        done
    else
        echo $shortname >> $unparsefile
        sort -u $unparsefile -o $unparsefile
        for backend in $backends; do
            unsuppfile="tests/conformance/unsupported-$backend.txt"
            sed --in-place "/^$shortname\$/d" $unsuppfile
        done
        echo "Unparseable: $shortname\n"
    fi
done

# Go over the unsupported files, see if any has become supported.
for backend in $backends; do
    unsuppfile="tests/conformance/unsupported-$backend.txt"
    unsupported=$(cat $unsuppfile)
    for shortname in $unsupported; do

        file=tests/wasm-tests/test/core/$shortname

        echo Trying to run: $shortname
        echo =============

        make $file.run-term TEST_CONCRETE_BACKEND=$backend
        if [ $? -eq 0 ]
        then
            # Now supported, remove.
            sed --in-place "/^$shortname\$/d" $unparsefile
            sed --in-place "/^$shortname\$/d" $unsuppfile
            sort -u $unsuppfile -o $unsuppfile
            echo "New supported ($backend): $shortname\n"
        else
            echo "Unsupported ($backend): $shortname\n"
        fi
    done
done
