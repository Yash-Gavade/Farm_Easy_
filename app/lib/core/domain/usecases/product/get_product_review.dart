import 'package:e_commerce_app/core/domain/entities/review/review.dart';
import 'package:e_commerce_app/core/domain/repositories/product_repository.dart';

class GetProductReview {
  final ProductRepository _repository;

  GetProductReview(this._repository);

  Future<List<Review>> execute({
    required String productId,
    String orderBy = 'created_at',
    bool descending = true,
  }) async {
    return await _repository.getProductReview(
      productId: productId,
      orderBy: orderBy,
      descending: descending,
    );
  }
}
