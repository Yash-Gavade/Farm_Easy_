import 'package:e_commerce_app/config/role_config.dart';

enum Flavor {
  user(1),
  admin(2);

  final int roleValue;
  const Flavor(this.roleValue);

  @override
  String toString() => "You logged in as ${name.toUpperCase()}";
}

class FlavorValues {
  final RoleConfig roleConfig;

  FlavorValues({
    required this.roleConfig,
  });
}

class FlavorConfig {
  static FlavorConfig? _instance;

  final Flavor flavor;
  final FlavorValues flavorValues;

  FlavorConfig({
    required this.flavor,
    required this.flavorValues,
  }) {
    _instance = this;
  }

  static FlavorConfig get instance {
    if (_instance == null) {
      throw Exception(
        "❌ FlavorConfig was not initialized. Make sure to call `FlavorConfig(...)` before `runApp()`.",
      );
    }
    return _instance!;
  }

  static bool get isAdmin => instance.flavor == Flavor.admin;
  static bool get isUser => instance.flavor == Flavor.user;
}