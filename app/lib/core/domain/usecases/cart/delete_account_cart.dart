import 'package:e_commerce_app/core/domain/repositories/cart_repository.dart';

class DeleteAccountCart {
  final CartRepository _repository;

  DeleteAccountCart(this._repository);

  Future<void> execute({required String accountId, required String cartId}) async {
    return _repository.deleteAccountCart(accountId: accountId, cartId: cartId);
  }
}
