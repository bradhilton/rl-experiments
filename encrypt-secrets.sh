#!/bin/bash

tar -cf secrets.tar .env .gcloud.json
gpg --symmetric --cipher-algo AES256 -o secrets.tar.gpg secrets.tar
rm secrets.tar