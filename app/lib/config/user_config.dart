import 'package:e_commerce_app/config/role_config.dart';
import 'package:e_commerce_app/themes/custom_bottom_sheet.dart';
import 'package:e_commerce_app/themes/custom_input_decoration.dart';
import 'package:e_commerce_app/themes/custom_text_theme.dart';
import 'package:flutter/material.dart';

class UserConfig implements RoleConfig {
  // TODO: Change App Name
  @override
  String appName() {
    return 'FarmAI Store';
  }

  // TODO: Change App Primary Color
  @override
  Color primaryColor() {
    return const Color(0xFF174351);
  }

  // TODO: Change App Primary Dark Color
  @override
  Color primaryDarkColor() {
    return const Color(0xFF102f39);
  }

  @override
  ThemeData theme() {
    return ThemeData(
      primaryColor: primaryColor(),
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryColor(),
      ),
      useMaterial3: true,
      textTheme: CustomTextTheme.textTheme,
      inputDecorationTheme: CustomInputDecoration.inputDecorationTheme,
      bottomSheetTheme: CustomBottomSheet.bottomSheetThemeData,
    );
  }

  @override
  ThemeData darkTheme() {
    return ThemeData(
      primaryColorDark: primaryDarkColor(),
      colorScheme: ColorScheme.fromSeed(
        seedColor: primaryDarkColor(),
        brightness: Brightness.dark,
      ),
      useMaterial3: true,
      textTheme: CustomTextTheme.textTheme,
      inputDecorationTheme: CustomInputDecoration.inputDecorationTheme,
      bottomSheetTheme: CustomBottomSheet.bottomSheetThemeData,
    );
  }
}
