import 'package:e_commerce_app/core/domain/entities/transaction/transaction.dart';
import 'package:e_commerce_app/core/domain/repositories/transaction_repository.dart';

class GetAllTransaction {
  final TransactionRepository _repository;

  GetAllTransaction(this._repository);

  Future<List<Transaction>> execute() async {
    return await _repository.getAllTransaction();
  }
}
