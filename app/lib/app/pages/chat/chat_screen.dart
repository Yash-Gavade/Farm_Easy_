import 'dart:io';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_chat_types/flutter_chat_types.dart' as types;
import 'package:flutter_chat_ui/flutter_chat_ui.dart';
import 'package:image_picker/image_picker.dart';
import 'package:uuid/uuid.dart';

import '../../../core/repositories/create chat_repository_impl.dart';


class ChatScreen extends StatefulWidget {
  final String sessionId;
  final String userId;
  final ChatRepositoryImpl repository;

  const ChatScreen({
    Key? key,
    required this.sessionId,
    required this.userId,
    required this.repository,
  }) : super(key: key);

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<types.Message> _messages = [];
  final _uuid = const Uuid();

  void _handleSendPressed(types.PartialText message) async {
    final userMessage = types.TextMessage(
      id: _uuid.v4(),
      author: types.User(id: widget.userId),
      text: message.text,
      createdAt: DateTime.now().millisecondsSinceEpoch,
    );

    setState(() => _messages.insert(0, userMessage));

    try {
      final botReply = await widget.repository.sendText(
        message.text,
      );

      final botMessage = types.TextMessage(
        id: _uuid.v4(),
        author: const types.User(id: 'bot',imageUrl: "assets/images/logo.png"),
        text: botReply,
        createdAt: DateTime.now().millisecondsSinceEpoch,
      );

      setState(() => _messages.insert(0, botMessage));
    } catch (e) {
      final errorMessage = types.TextMessage(
        id: _uuid.v4(),
        author: const types.User(id: 'bot'),
        text: '❌ Failed to get response.',
        createdAt: DateTime.now().millisecondsSinceEpoch,
      );

      setState(() => _messages.insert(0, errorMessage));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('FarmAI BOT')),
      body: Chat(
        messages: _messages,
        onSendPressed: _handleSendPressed,
        user: types.User(id: widget.userId),
        showUserAvatars: true,
        showUserNames: true,
      ),
    );
  }
}
