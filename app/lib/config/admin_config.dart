import 'package:e_commerce_app/config/role_config.dart';
import 'package:e_commerce_app/themes/custom_bottom_sheet.dart';
import 'package:e_commerce_app/themes/custom_text_theme.dart';
import 'package:flutter/material.dart';

import '../themes/custom_input_decoration.dart';

class AdminConfig implements RoleConfig {
  // TODO: Change Admin App Name
  @override
  String appName() {
    return 'FarmAI Vendor';
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
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: primaryColor(),
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
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: primaryDarkColor(),
      ),
      useMaterial3: true,
      textTheme: CustomTextTheme.textTheme,
      inputDecorationTheme: CustomInputDecoration.inputDecorationTheme,
      bottomSheetTheme: CustomBottomSheet.bottomSheetThemeData,
    );
  }
}
