import 'package:e_commerce_app/core/domain/entities/cart/cart.dart';
import 'package:e_commerce_app/core/domain/repositories/cart_repository.dart';

class UpdateAccountCart {
  final CartRepository _repository;

  UpdateAccountCart(this._repository);

  Future<void> execute({required String accountId, required Cart data}) async {
    return _repository.updateAccountCart(accountId: accountId, data: data);
  }
}
