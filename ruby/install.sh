#!/bin/bash

sudo apt install ruby ruby-dev -y
apt install -y git ruby gcc ruby-dev rake make
gem install specific_install
gem specific_install https://github.com/ruby-numo/numo-narray.git
