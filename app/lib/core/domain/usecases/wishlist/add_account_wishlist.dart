import 'package:e_commerce_app/core/domain/entities/wishlist/wishlist.dart';
import 'package:e_commerce_app/core/domain/repositories/wishlist_repository.dart';

class AddAccountWishlist {
  final WishlistRepository _repository;

  AddAccountWishlist(this._repository);

  Future<void> execute({
    required String accountId,
    required Wishlist data,
  }) async {
    return _repository.addAccountWishlist(accountId: accountId, wishlist: data);
  }
}
