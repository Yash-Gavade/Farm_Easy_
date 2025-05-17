import 'package:e_commerce_app/core/domain/entities/account/account.dart';
import 'package:e_commerce_app/core/domain/entities/cart/cart.dart';
import 'package:e_commerce_app/core/domain/entities/transaction/transaction.dart';

abstract class CheckoutRepository {
  Transaction startCheckout({
    required List<Cart> cart,
    required Account account,
  });

  Future<void> pay({
    required Transaction data,
  });
}
