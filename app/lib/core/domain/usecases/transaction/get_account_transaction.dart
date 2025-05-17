import 'package:e_commerce_app/core/domain/entities/transaction/transaction.dart';
import 'package:e_commerce_app/core/domain/repositories/transaction_repository.dart';

class GetAccountTransaction {
  final TransactionRepository _repository;

  GetAccountTransaction(this._repository);

  Future<List<Transaction>> execute({
    required String accountId,
  }) async {
    return await _repository.getAccountTransaction(
      accountId: accountId,
    );
  }
}
