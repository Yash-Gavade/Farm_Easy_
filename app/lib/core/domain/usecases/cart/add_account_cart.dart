import 'package:e_commerce_app/core/domain/entities/cart/cart.dart';
import 'package:e_commerce_app/core/domain/repositories/cart_repository.dart';

class AddAccountCart {
  final CartRepository _repository;

  AddAccountCart(this._repository);

  Future<void> execute({required String accountId, required Cart data}) async {
    return _repository.addAccountCart(accountId: accountId, data: data);
  }
}
