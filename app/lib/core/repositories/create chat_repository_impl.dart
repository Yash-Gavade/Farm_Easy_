import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:e_commerce_app/app/constants/collections_name.dart';

class ChatRepositoryImpl {
  final String baseUrl;

  ChatRepositoryImpl({required this.baseUrl});

  Future<String> sendText(String question) async {
    try {
      final uri = Uri.parse('https://d002-2a02-3035-66b-55a1-81ab-dfe2-fbad-ed6.ngrok-free.app/ask/');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'question': question}),
      );

      if (response.statusCode == 200) {
        final jsonData = jsonDecode(response.body);
        return jsonData['answer'] ?? 'No answer found.';
      } else {
        print('❌ Error: ${response.statusCode} - ${response.body}');
        return '❌ Failed to get response.';
      }
    } catch (e) {
      print('❌ Exception: $e');
      return '❌ Failed to get response.';
    }
  }
}