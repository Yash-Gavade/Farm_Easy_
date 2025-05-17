import 'package:e_commerce_app/app/providers/account_provider.dart';
import 'package:e_commerce_app/app/providers/address_provider.dart';
import 'package:e_commerce_app/app/widgets/radio_card.dart';
import 'package:e_commerce_app/core/domain/entities/address/address.dart';
import 'package:e_commerce_app/routes.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:provider/provider.dart';

class ManageAddressPage extends StatefulWidget {
  const ManageAddressPage({super.key});

  @override
  State<ManageAddressPage> createState() => _ManageAddressPageState();
}

class _ManageAddressPageState extends State<ManageAddressPage> {
  bool returnAddress = false;

  String accountId = FirebaseAuth.instance.currentUser!.uid;

  Address? selectedAddress;

  bool _isLoading = false;

  @override
  void initState() {
    SchedulerBinding.instance.addPostFrameCallback(
      (_) {
        returnAddress = ModalRoute.of(context)!.settings.arguments as bool;

        selectedAddress =
            context.read<AccountProvider>().account.primaryAddress;
        context.read<AddressProvider>().getAddress(accountId: accountId);
      },
    );
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Manage Address'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pushNamed(RouteName.kAddAddress);
            },
            child: const Text('Add Address'),
          ),
        ],
      ),
      body: Consumer<AddressProvider>(
        builder: (context, value, child) {
          if (value.isLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }

          return Column(
            children: [
              // Address List
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 16,
                  ),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: MediaQuery.of(context).size.height * 0.5,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (value.listAddress.isEmpty)
                          Center(
                            child: Text(
                              'Address is Empty,\nAdd your address for shipping purpose',
                              style: Theme.of(context).textTheme.bodyMedium,
                              textAlign: TextAlign.center,
                            ),
                          ),
                        if (value.listAddress.isNotEmpty)
                          ...value.listAddress.map(
                                (e) => Padding(
                              padding: const EdgeInsets.only(bottom: 16.0),
                              child: RadioCard<Address>(
                                title: e.name,
                                subtitle: '${e.address} ${e.city} ${e.zipCode}',
                                value: e,
                                selectedValue: selectedAddress,
                                onChanged: (value) {
                                  setState(() {
                                    selectedAddress = value;
                                  });
                                },
                                onDelete: () {
                                  value.delete(
                                      accountId: accountId, addressId: e.addressId);
                                  if (selectedAddress == e) {
                                    selectedAddress = null;
                                    context.read<AccountProvider>().update(
                                      data: {'primary_address': null},
                                    );
                                  }
                                },
                                onEdit: () {
                                  NavigateRoute.toEditAddress(
                                      context: context, address: e);
                                },
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),

              // Button
              if (value.listAddress.isNotEmpty)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 16,
                  ),
                  child: ElevatedButton(
                    onPressed: selectedAddress == null
                        ? null
                        : () async {
                            if (_isLoading) return;

                            if (returnAddress) {
                              Navigator.of(context).pop(selectedAddress);
                              return;
                            }

                            setState(() {
                              _isLoading = true;
                            });

                            Address? oldData = context
                                .read<AccountProvider>()
                                .account
                                .primaryAddress;
                            await value
                                .changePrimary(
                                    accountId: accountId,
                                    data: selectedAddress!,
                                    oldData: oldData)
                                .whenComplete(
                              () async {
                                if (!context.mounted) return;
                                context.read<AccountProvider>().getProfile();
                                setState(() {
                                  _isLoading = false;
                                });
                                Navigator.of(context).pop();
                              },
                            );
                          },
                    child: _isLoading
                        ? const Center(
                            child: SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                              ),
                            ),
                          )
                        : returnAddress
                            ? const Text('Select Address')
                            : const Text('Change Primary Address'),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}
