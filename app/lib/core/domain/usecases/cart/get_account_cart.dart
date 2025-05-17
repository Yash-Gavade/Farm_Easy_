import 'package:e_commerce_app/core/domain/entities/cart/cart.dart';
import 'package:e_commerce_app/core/domain/repositories/cart_repository.dart';

class GetAccountCart {
  final CartRepository _repository;

  GetAccountCart(this._repository);

  Future<List<Cart>> execute({required String accountId}) async {
    return _repository.getAccountCart(accountId: accountId);
  }
}
