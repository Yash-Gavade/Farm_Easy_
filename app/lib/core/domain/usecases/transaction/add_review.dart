import 'package:e_commerce_app/core/domain/entities/review/review.dart';
import 'package:e_commerce_app/core/domain/repositories/transaction_repository.dart';

class AddReview {
  final TransactionRepository _repository;

  AddReview(this._repository);

  Future<void> execute({required String transactionId, required Review data}) async {
    return await _repository.addReview(
      transactionId: transactionId,
      data: data,
    );
  }
}
