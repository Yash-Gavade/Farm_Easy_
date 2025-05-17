import 'package:e_commerce_app/core/domain/repositories/auth_repository.dart';

class LogoutAccount {
  final AuthRepository _repository;

  LogoutAccount(this._repository);

  Future<void> execute() async {
    await _repository.logout();
  }
}
