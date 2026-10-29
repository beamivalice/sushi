import Foundation
import Security

struct ConnectionProfile: Codable, Identifiable, Equatable {
    var id = UUID().uuidString
    var name:String
    var endpoint:String
    var binary:String
    var arguments:String
    var rememberKey = false
    func validate() throws {
        _ = try Connection(endpoint:endpoint).request("health")
        _ = try LaunchConfiguration.decode(arguments)
    }
}
struct CredentialStore {
    private let service = "org.sushi.SushiControl.connection"
    func read(_ id:String) -> String? {
        let query:[String:Any] = [kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:id,kSecReturnData as String:true,kSecMatchLimit as String:kSecMatchLimitOne]
        var result:CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary,&result) == errSecSuccess, let data = result as? Data else { return nil }
        return String(data:data,encoding:.utf8)
    }
    func write(_ key:String,id:String) throws {
        let query:[String:Any] = [kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:id]
        if key.isEmpty { try remove(id); return }
        let data = Data(key.utf8)
        var status = SecItemUpdate(query as CFDictionary,[kSecValueData as String:data] as CFDictionary)
        if status == errSecItemNotFound {
            var item = query; item[kSecValueData as String] = data
            item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            status = SecItemAdd(item as CFDictionary,nil)
        }
        guard status == errSecSuccess else { throw ServerFailure(message:"Could not save the API key in Keychain (\(status)).") }
    }
    func remove(_ id:String) throws {
        let status = SecItemDelete([kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:id] as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw ServerFailure(message:"Could not remove the saved API key (\(status)).") }
    }
}
