import 'package:egrocer_seller/helper/utils/generalImports.dart';
import 'package:flutter/material.dart';

class ActionButtonWidget extends StatelessWidget {
  final String buttonName;
  final VoidCallback voidCallback;
  final EdgeInsetsGeometry? padding;

  ActionButtonWidget(
      {super.key,
      required this.buttonName,
      required this.voidCallback,
      this.padding});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: padding ?? EdgeInsets.zero,
      child: GestureDetector(
        onTap: voidCallback,
        child: Container(
          height: 45,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: ColorsRes.appColor,
            borderRadius: BorderRadius.circular(
              7,
            ),
          ),
          child: Text(
            getTranslatedValue(context, buttonName),
            softWrap: true,
            style: TextStyle(
              color: Colors.white,
            ),
          ),
        ),
      ),
    );
  }
}
