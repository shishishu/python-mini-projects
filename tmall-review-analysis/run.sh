#!/usr/bin/env bash
# get shop_name
# shop_name="chabiubiu"
if [ $# != 1 ]
then
    echo "error: shop name is required, please pass the variable in the command ..."
    exit 1
fi

shop_name=$1
echo "shop name is: "${shop_name}

echo "start parse html..."
python parse_html.py --shop_name ${shop_name}

echo "start review analysis..."
python review_analysis.py --shop_name ${shop_name}

echo "all the tasks are finished..."