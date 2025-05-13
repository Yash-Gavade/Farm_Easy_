import 'package:egrocer_seller/helper/utils/generalImports.dart';

import 'package:flutter/material.dart';
getOrderStatusContainer({
  required BuildContext context,
  required String svgIconName,
  required String title,
  required bool isActive,
}) {
  return Container(
    height: 45,
    decoration: BoxDecoration(
      color: isActive ? ColorsRes.appColor : Theme.of(context).cardColor,
      borderRadius: BorderRadius.circular(10),
    ),
    margin: EdgeInsetsDirectional.only(start: 10, bottom: 10, top: 10),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        Widgets.getSizedBox(width: 10),
        Widgets.defaultImg(
            image: svgIconName,
            iconColor:
                isActive ? ColorsRes.appColorWhite : ColorsRes.mainTextColor,
            width: 20,
            height: 20),
        Widgets.getSizedBox(width: 10),
        Text(
          title,
          softWrap: true,
          textAlign: TextAlign.center,
          style: TextStyle(
              color: isActive
                  ? ColorsRes.appColorWhite
                  : ColorsRes.mainTextColor,
              fontWeight: FontWeight.w400),
        ),
        Widgets.getSizedBox(width: 10),
      ],
    ),
  );
}
