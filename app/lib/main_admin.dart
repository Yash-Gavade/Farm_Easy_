import 'package:e_commerce_app/app/app.dart';
import 'package:e_commerce_app/config/admin_config.dart';
import 'package:e_commerce_app/config/flavor_config.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'firebase_options.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
  } on FirebaseException catch (e) {
    if (e.code == 'duplicate-app') {
      print("⚠️ Firebase app already exists, ignoring.");
    } else {
      rethrow; // Re-throw unexpected errors
    }
  }

  FlavorConfig(
    flavor: Flavor.admin,
    flavorValues: FlavorValues(roleConfig: AdminConfig()),
  );

  runApp(const App());
}
